#!/usr/bin/env bash
set -euo pipefail

python3 scripts/run_hank.py --output-dir outputs/hank_core
python3 experiments/exp00_hank_core_audit.py --hank-core-dir outputs/hank_core
python3 experiments/final/01_validate_ssj.py
python3 experiments/final/03_estimate_information_value.py
python3 experiments/final/04_mechanism_checks.py
python3 experiments/final/05_lqg_benchmark.py
python3 experiments/final/06_feedback_rate_check.py

(
  cd article
  pdflatex -interaction=nonstopmode main.tex
  bibtex main
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
)
