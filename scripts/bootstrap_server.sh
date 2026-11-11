#!/usr/bin/env bash
# Run in the uploaded repository: bash scripts/bootstrap_server.sh [--check-only]
# Creates a project venv only. Does not change drivers or install OS packages.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
project_dir="$(pwd -P)"
mkdir -p runs/setup models data/raw data/normalized
date -u '+%Y-%m-%dT%H:%M:%SZ' > runs/setup/started-utc.txt
uname -a > runs/setup/system.txt
nvidia-smi > runs/setup/nvidia-smi.txt
nvidia-smi topo -m > runs/setup/gpu-topology.txt
free -h > runs/setup/memory.txt
df -h "$project_dir" > runs/setup/disk.txt
python_bin="${TABLEQA_PYTHON:-python3}"
"$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12 for these initial pins"; print(sys.version)'
if [[ "${1:-}" == "--check-only" ]]; then
  cat runs/setup/nvidia-smi.txt runs/setup/memory.txt runs/setup/disk.txt
  exit 0
fi
if [[ -e .venv && ! -f .venv/pyvenv.cfg ]]; then
  echo '.venv exists but is not a venv; refusing to overwrite' >&2
  exit 1
fi
if [[ ! -f .venv/pyvenv.cfg ]]; then
  "$python_bin" -m venv .venv
fi
source .venv/bin/activate
# Keep the bundled pip; upgrading it first can stall on an unreachable CDN.
# cu121 wheel installation follows the official PyTorch previous-versions page.
python -m pip install 'torch==2.5.1' --index-url "${TABLEQA_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
python -m pip install -r requirements-server.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pip freeze > runs/setup/environment.lock.txt
export HF_HOME="$project_dir/models/huggingface"
printf 'export HF_HOME=%q\n' "$HF_HOME" > runs/setup/env.sh
python -c 'import torch; print("torch",torch.__version__,"runtime",torch.version.cuda); assert torch.cuda.is_available(); print("GPUs",torch.cuda.device_count()); assert torch.cuda.device_count() >= 2; print("BF16",torch.cuda.is_bf16_supported()); assert torch.cuda.is_bf16_supported()'
python -m unittest discover -s tests -v 2>&1 | tee runs/setup/tests.log
python -m tableqa demo --output runs/setup/demo
echo 'Environment installed; model/data downloads and real experiments have NOT started.'
