#!/bin/bash
set -euo pipefail
[[ -f configs/mitigation.suite.en.v2.json ]] || { echo 'Run from the repository root' >&2; exit 2; }
python scripts/submit_mitigation_suite.py --submit "$@"
