#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
unset OUTPUT_DIR

python3 "$SCRIPT_DIR/train_grpo.py"
