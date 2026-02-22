#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

echo "Running Bell example..."
python -m qsim.cli --circuit circuits/bell.txt --shots 2000 --seed 1 --check-norm

echo ""
echo "Running GHZ(4) example..."
python -m qsim.cli --circuit circuits/ghz4.txt --shots 2000 --seed 2 --check-norm

echo ""
echo "Running benchmark..."
python -m qsim.cli --benchmark --seed 3