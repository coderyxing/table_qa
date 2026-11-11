#!/usr/bin/env bash
set -euo pipefail
# Read-only diagnostics: no installation, driver changes or model downloads.
nvidia-smi
nvidia-smi topo -m
free -h
df -h .
python -c 'import sys, torch; print(sys.version); print("torch",torch.__version__,"runtime",torch.version.cuda); print("devices",torch.cuda.device_count()); print("bf16",torch.cuda.is_bf16_supported())'
python -m pip freeze
