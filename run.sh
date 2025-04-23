#!/usr/bin/env bash
set -euo pipefail
python build_modality.py "$@"
python main.py "$@"
