from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_length_controlled_halueval as benchmark_cli

from hallu_llm.controlled_benchmark import (
    CandidateConstraints,
    build_review_rows,
    candidate_diagnostics,
    finalize_reviewed_proposals,
    generate_proposal,
    length_bounds,
    parse_generated_proposal,
    reconcile_dual_reviews,
    summarize_dual_reviews,
    summarize_reviews,
)
from hallu_llm.data import QASample, load_halueval_qa
from hallu_llm.detection import stable_sample_id


class FakeGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            text=next(self.outputs),
            mean_confidence=0.8,
            mean_entropy=0.2,
        )


class ControlledBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.constraints = CandidateConstraints(
            max_relative_length_difference=0.25,
            max_sentence_count_difference=0,
        )
        self.sample = QASample(
            0,
            "What is the capital?",
            "Paris is the capital of France.",
            "Paris",
            "Paris is in Germany.",
            sample_id="source-a",
        )

    def _two_proposals(self) -> list[dict[str, object]]:
        second_sample = QASample(
            1,
            "Which city is the capital of Italy?",
            "Rome is the capital of Italy.",
            "Rome",
            "Milan",
            sample_id="source-b",
        )
        return [
            generate_proposal(
                FakeGenerator(
                    ['{"candidate":"Lyon","changed_claim":"city"}']
                ),
                self.sample,
                self.constraints,
                max_attempts=1,
                max_new_tokens=32,
                temperature=0.0,
                top_p=1.0,
            ),
            generate_proposal(
                FakeGenerator(
                    ['{"candidate":"Milan","changed_claim":"city"}']
                ),
                second_sample,
                self.constraints,
                max_attempts=1,
                max_new_tokens=32,
                temperature=0.0,
                top_p=1.0,
            ),
        ]

    def test_diagnostics_enforce_length_and_difference(self) -> None:
        valid = candidate_diagnostics("Paris", "Lyon", self.constraints)
        same = candidate_diagnostics("Paris", "Paris", self.constraints)
        long = candidate_diagnostics(
            "Paris",
            "A very long unsupported candidate",
            self.constraints,
        )

        self.assertTrue(valid["structurally_valid"])
        self.assertIn("same_as_grounded", same["issues"])
        self.assertIn("length_out_of_range", long["issues"])
        self.assertEqual(length_bounds("Paris", 0.25), (4, 6))

    def test_diagnostics_match_answer_style_and_entity_count(self) -> None:
        style = candidate_diagnostics("Paris", "Lyon.", self.constraints)
        entities = candidate_diagnostics(
            "Paris met France",
            "London met locals",
            self.constraints,
        )

        self.assertIn("answer_style_mismatch", style["issues"])
        self.assertIn("named_entity_count_mismatch", entities["issues"])

    def test_parses_json_and_label_fallback(self) -> None:
        fence = chr(96) * 3
        self.assertEqual(
            parse_generated_proposal(
                fence
                + 'json\n{"candidate":"Lyon","changed_claim":"city"}\n'
                + fence
            ),
            ("Lyon", "city"),
        )
        self.assertEqual(
            parse_generated_proposal(
                "CANDIDATE: Lyon\nCHANGE: Replaced the city."
            ),
            ("Lyon", "Replaced the city."),
        )

    def test_generation_retries_failed_mechanical_candidate(self) -> None:
        generator = FakeGenerator(
            [
                '{"candidate":"Paris","changed_claim":"none"}',
                '{"candidate":"Lyon","changed_claim":"city"}',
            ]
        )

        proposal = generate_proposal(
            generator,
            self.sample,
            self.constraints,
            max_attempts=3,
            max_new_tokens=32,
            temperature=0.8,
            top_p=0.9,
        )

        self.assertEqual(proposal["proposed_hallucinated_answer"], "Lyon")
        self.assertEqual(proposal["selected_attempt"], 2)
        self.assertEqual(len(proposal["attempts"]), 2)

    def test_review_gate_requires_reason_and_rechecks_override(self) -> None:
        proposal = generate_proposal(
            FakeGenerator(
                ['{"candidate":"Lyon","changed_claim":"city"}']
            ),
            self.sample,
            self.constraints,
            max_attempts=1,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
        )

        with self.assertRaisesRegex(ValueError, "requires a review_reason"):
            finalize_reviewed_proposals(
                [proposal],
                {"source-a": {"review_status": "approved"}},
                self.constraints,
            )
        with self.assertRaisesRegex(ValueError, "requires a reviewer"):
            finalize_reviewed_proposals(
                [proposal],
                {
                    "source-a": {
                        "review_status": "approved",
                        "review_reason": "Fact checked.",
                    }
                },
                self.constraints,
            )
        with self.assertRaisesRegex(ValueError, "fails structural checks"):
            finalize_reviewed_proposals(
                [proposal],
                {
                    "source-a": {
                        "review_status": "approved",
                        "reviewed_candidate": "A very long unsupported candidate",
                        "review_reason": "Fact checked.",
                        "reviewer": "reviewer-1",
                    }
                },
                self.constraints,
            )

    def test_approved_review_preserves_source_id_and_freezes_candidate(self) -> None:
        proposal = generate_proposal(
            FakeGenerator(
                ['{"candidate":"Lyon","changed_claim":"city"}']
            ),
            self.sample,
            self.constraints,
            max_attempts=1,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
        )
        reviews = {
            "source-a": {
                "review_status": "approved",
                "review_reason": "Lyon is contradicted by the evidence.",
                "reviewer": "reviewer-1",
            }
        }

        final_rows, audit_rows = finalize_reviewed_proposals(
            [proposal], reviews, self.constraints
        )
        quality = summarize_reviews(
            [proposal],
            audit_rows,
            minimum_approved_samples=1,
            require_all_reviewed=True,
        )

        self.assertEqual(final_rows[0]["sample_id"], "source-a")
        self.assertEqual(final_rows[0]["hallucinated_answer"], "Lyon")
        self.assertTrue(quality["ready_for_detection"])
        self.assertEqual(build_review_rows([proposal])[0]["review_status"], "pending")

    def test_dual_review_agreement_finalizes_without_adjudication(self) -> None:
        proposal = generate_proposal(
            FakeGenerator(
                ['{"candidate":"Lyon","changed_claim":"city"}']
            ),
            self.sample,
            self.constraints,
            max_attempts=1,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
        )
        reviews = {
            "reviewer_1": {
                "source-a": {
                    "review_status": "approved",
                    "review_reason": "Unsupported city.",
                    "reviewer": "Alice",
                }
            },
            "reviewer_2": {
                "source-a": {
                    "review_status": "approved",
                    "review_reason": "Contradicted by the evidence.",
                    "reviewer": "Bob",
                }
            },
        }

        final_rows, audit_rows, adjudication_rows = reconcile_dual_reviews(
            [proposal],
            reviews,
            {},
            self.constraints,
        )
        quality = summarize_dual_reviews(
            [proposal],
            audit_rows,
            minimum_approved_samples=1,
            require_all_reviewed=True,
        )

        self.assertEqual(len(final_rows), 1)
        self.assertEqual(adjudication_rows, [])
        self.assertTrue(quality["ready_for_detection"])
        self.assertEqual(quality["cohen_kappa_status"], 1.0)

    def test_dual_review_disagreement_requires_adjudication(self) -> None:
        proposal = generate_proposal(
            FakeGenerator(
                ['{"candidate":"Lyon","changed_claim":"city"}']
            ),
            self.sample,
            self.constraints,
            max_attempts=1,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
        )
        reviews = {
            "reviewer_1": {
                "source-a": {
                    "review_status": "approved",
                    "review_reason": "Unsupported city.",
                    "reviewer": "Alice",
                }
            },
            "reviewer_2": {
                "source-a": {
                    "review_status": "rejected",
                    "review_reason": "Needs a clearer edit.",
                    "reviewer": "Bob",
                }
            },
        }

        final_rows, audit_rows, adjudication_rows = reconcile_dual_reviews(
            [proposal],
            reviews,
            {},
            self.constraints,
        )
        self.assertEqual(final_rows, [])
        self.assertEqual(
            audit_rows[0]["final_review_status"],
            "adjudication_pending",
        )
        self.assertEqual(len(adjudication_rows), 1)

        adjudications = {
            "source-a": {
                "review_status": "approved",
                "reviewed_candidate": "Lyon",
                "review_reason": "The single city substitution is valid.",
                "reviewer": "Carol",
            }
        }
        final_rows, audit_rows, _ = reconcile_dual_reviews(
            [proposal],
            reviews,
            adjudications,
            self.constraints,
        )
        self.assertEqual(len(final_rows), 1)
        self.assertTrue(audit_rows[0]["adjudicated"])

    def test_dual_review_requires_distinct_reviewer_names(self) -> None:
        proposal = generate_proposal(
            FakeGenerator(
                ['{"candidate":"Lyon","changed_claim":"city"}']
            ),
            self.sample,
            self.constraints,
            max_attempts=1,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
        )
        decision = {
            "source-a": {
                "review_status": "approved",
                "review_reason": "Unsupported city.",
                "reviewer": "Same Person",
            }
        }
        with self.assertRaisesRegex(ValueError, "two distinct"):
            reconcile_dual_reviews(
                [proposal],
                {"reviewer_1": decision, "reviewer_2": decision},
                {},
                self.constraints,
            )

    def test_local_jsonl_loader_preserves_frozen_ids(self) -> None:
        rows = [
            {
                "sample_id": "source-a",
                "question": "Question",
                "knowledge": "Knowledge",
                "right_answer": "Right",
                "hallucinated_answer": "Wrong",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controlled.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            samples = load_halueval_qa(
                num_samples=None,
                seed=7,
                dataset_cfg={"path": str(path), "shuffle": False},
            )

        self.assertEqual(samples[0].sample_id, "source-a")
        self.assertEqual(stable_sample_id(samples[0]), "source-a")

    def test_local_jsonl_rejects_duplicate_ids(self) -> None:
        row = {
            "sample_id": "duplicate",
            "question": "Question",
            "knowledge": "Knowledge",
            "right_answer": "Right",
            "hallucinated_answer": "Wrong",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controlled.jsonl"
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
                load_halueval_qa(
                    num_samples=None,
                    seed=7,
                    dataset_cfg={"path": str(path), "shuffle": False},
                )

    def test_finalize_cli_writes_dataset_and_frozen_split(self) -> None:
        proposals = self._two_proposals()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_name = "integration"
            experiment_dir = root / experiment_name
            experiment_dir.mkdir()
            proposals_path = experiment_dir / "proposals.jsonl"
            proposals_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in proposals
                ),
                encoding="utf-8",
            )

            review_rows = build_review_rows(proposals)
            for row in review_rows:
                row["review_status"] = "approved"
                row["review_reason"] = "Unsupported city verified manually."
                row["reviewer"] = "test-reviewer"
            review_path = experiment_dir / "proposal_review.csv"
            with review_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(review_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(review_rows)

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_name": experiment_name,
                        "output_dir": str(root),
                        "constraints": {
                            "max_relative_length_difference": 0.25,
                            "max_sentence_count_difference": 0,
                            "forbid_newlines": True,
                        },
                        "review": {
                            "file": "proposal_review.csv",
                            "minimum_approved_samples": 2,
                            "require_all_reviewed": True,
                        },
                        "split": {
                            "dev_fraction": 0.5,
                            "seed": 20260901,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                sys,
                "argv",
                [
                    "build_length_controlled_halueval.py",
                    "--config",
                    str(config_path),
                    "--finalize-only",
                ],
            ):
                benchmark_cli.main()

            final_rows = benchmark_cli._read_jsonl(
                experiment_dir / "controlled_halueval.jsonl"
            )
            with (experiment_dir / "split_manifest.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                split_rows = list(csv.DictReader(handle))

        self.assertEqual(
            [row["sample_id"] for row in final_rows],
            ["source-a", "source-b"],
        )
        self.assertEqual(len(split_rows), 2)
        self.assertEqual(
            {row["split"] for row in split_rows},
            {"development", "test"},
        )

    def test_dual_review_cli_finalizes_tracked_protocol_shape(self) -> None:
        proposals = self._two_proposals()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_name = "dual-integration"
            experiment_dir = root / experiment_name
            experiment_dir.mkdir()
            (experiment_dir / "proposals.jsonl").write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in proposals
                ),
                encoding="utf-8",
            )

            review_files = ["review_1.csv", "review_2.csv"]
            for review_file, reviewer in zip(
                review_files,
                ("Alice", "Bob"),
            ):
                rows = build_review_rows(proposals)
                for row in rows:
                    row["review_status"] = "approved"
                    row["review_reason"] = "Unsupported city verified."
                    row["reviewer"] = reviewer
                with (experiment_dir / review_file).open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=list(rows[0].keys())
                    )
                    writer.writeheader()
                    writer.writerows(rows)

            config_path = root / "dual_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_name": experiment_name,
                        "output_dir": str(root),
                        "constraints": {
                            "max_relative_length_difference": 0.25,
                            "max_sentence_count_difference": 0,
                            "max_named_entity_count_difference": 0,
                            "require_answer_style_match": True,
                            "forbid_newlines": True,
                        },
                        "review": {
                            "files": review_files,
                            "adjudication_file": "adjudication.csv",
                            "require_distinct_reviewers": True,
                            "minimum_approved_samples": 2,
                            "require_all_reviewed": True,
                        },
                        "split": {
                            "dev_fraction": 0.5,
                            "seed": 20260901,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                sys,
                "argv",
                [
                    "build_length_controlled_halueval.py",
                    "--config",
                    str(config_path),
                    "--finalize-only",
                ],
            ):
                benchmark_cli.main()

            quality = json.loads(
                (experiment_dir / "quality.json").read_text(encoding="utf-8")
            )
            final_rows = benchmark_cli._read_jsonl(
                experiment_dir / "controlled_halueval.jsonl"
            )

        self.assertTrue(quality["ready_for_detection"])
        self.assertEqual(
            quality["review_protocol"],
            "dual_independent_review_with_adjudication",
        )
        self.assertEqual(len(final_rows), 2)

    def test_review_csv_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal_review.csv"
            path.write_text(
                "sample_id,review_status\nsource-a,pending\nsource-a,pending\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                benchmark_cli._read_review_csv(path)

    def test_proposal_fingerprint_and_seed_cover_generation_settings(self) -> None:
        first = benchmark_cli._proposal_fingerprint(
            [self.sample],
            {"path": "model"},
            {"temperature": 0.8},
            {"max_relative_length_difference": 0.25},
            7,
        )
        second = benchmark_cli._proposal_fingerprint(
            [self.sample],
            {"path": "model"},
            {"temperature": 0.9},
            {"max_relative_length_difference": 0.25},
            7,
        )

        self.assertNotEqual(first, second)
        self.assertEqual(
            benchmark_cli._sample_generation_seed(7, "source-a"),
            benchmark_cli._sample_generation_seed(7, "source-a"),
        )
        self.assertNotEqual(
            benchmark_cli._sample_generation_seed(7, "source-a"),
            benchmark_cli._sample_generation_seed(7, "source-b"),
        )


if __name__ == "__main__":
    unittest.main()
