# Project dataset inventory

Prepared locally on 2026-09-16, following `configs/benchmark_suite_protocol.json`
and `configs/controlled_confirmatory_protocol.json`.

| Dataset | Planned use | Local preparation |
| --- | --- | --- |
| HaluEval-QA | Full English paired hallucination detection | Ready: 10,000 pairs (20,000 candidate decisions) |
| HalluLens | PreciseWikiQA, LongWiki, MixedEntities, GeneratedEntities | Official repository downloaded; external assets pending |
| RAGTruth | Full good-quality test response and span detection | Ready: 2,675 responses; 25 quality-flagged responses excluded |
| Mu-SHROOM | Primary multilingual span detection | Ready: 1,902 test responses, 14 languages; 499 validation responses stored separately |
| TruthfulQA | Separate factuality control | Ready: all 790 questions in the downloaded official CSV; project binary-choice adaptation |
| Controlled HaluEval | Focused English/Chinese confirmatory study | Pending generation of 550 proposals, two independent reviews, at least 500 approved pairs, and audited Chinese translation |

## Files

Normalized datasets and quality reports are in `data/benchmarks/prepared/`.
`manifest.json` records source revisions, SHA-256 hashes, counts, exclusions and
readiness. Raw Git repositories are in `data/benchmarks/raw/`; Hugging Face cache
is in `data/cache/huggingface/`. These data directories are ignored by Git.
The local dataset-loader dependencies are in `.dataset-deps/`.

Mu-SHROOM's `mu_shroom.jsonl` contains only test rows. Development rows are in
`mu_shroom.validation.jsonl`. Never pool validation into final test results.
TruthfulQA is reported separately, and task-specific benchmark metrics are not
averaged into one overall score.

## Remaining preparation

HalluLens requires GoodWiki with WikiRank scores, the Wikipedia SQLite dump and
title database, medicine data, and ITIS hierarchy/longnames files. The upstream
downloader uses Unix shell commands. Run its preparation on the Linux cluster
after installing its upstream dependencies, from the project root:

```bash
python scripts/prepare_benchmark_suite.py --benchmark hallulens --download --download-hallulens-data
```

The manifest lists missing assets and keeps `full_suite_ready` false until they
exist. Downloading the repository alone does not complete HalluLens preparation.
Dynamic task execution also requires the upstream model endpoint and, for
generated entities, search/judge credentials.

For the controlled study, follow the length-controlled workflow in `README.md`:
generate proposals with `configs/controlled_benchmark.full.json`, obtain both
independent reviews and any adjudication, finalize the frozen development/test
split, and run translation quality checks. Do not label unreviewed proposals
as the final dataset. The earlier 20-sample smoke and 150-sample translation
preflight are HaluEval subsets, not additional independent benchmarks.

## Validation

Preparation recorded no rejected HaluEval rows and no span-alignment warnings
for RAGTruth or Mu-SHROOM. The benchmark unit-test command encountered Windows
error 1455 while importing PyTorch (insufficient paging-file space), before
tests could execute. See the manifest and quality JSON files for data checks.
