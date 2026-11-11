#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for attempt in $(seq 1 120); do
  if [[ -f runs/wheels/DOWNLOAD_COMPLETE && -f runs/cuda-wheels/DOWNLOAD_COMPLETE ]]; then break; fi
  sleep 10
done
test -f runs/wheels/DOWNLOAD_COMPLETE
test -f runs/cuda-wheels/DOWNLOAD_COMPLETE
source .venv/bin/activate
python scripts/pip_ipv4.py install --no-index --no-deps runs/cuda-wheels/*.whl
python scripts/pip_ipv4.py install runs/wheels/torch-2.5.1-cp312-cp312-manylinux1_x86_64.whl --find-links runs/cuda-wheels --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 20 --retries 2 --disable-pip-version-check
python scripts/pip_ipv4.py install -r requirements-server.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 20 --retries 2 --disable-pip-version-check
python scripts/pip_ipv4.py install 'setuptools>=68' wheel --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 20 --retries 2 --disable-pip-version-check
python scripts/pip_ipv4.py install --no-deps --no-build-isolation -e .
python -m pip check
python -m pip freeze > runs/setup/environment.lock.txt
python -c 'import torch; print("torch",torch.__version__,"runtime",torch.version.cuda); assert torch.cuda.is_available(); x=torch.ones(4,device="cuda",dtype=torch.bfloat16); print("GPU",torch.cuda.get_device_name(),"sum",x.sum().item())'
python -m unittest discover -s tests -v 2>&1 | tee runs/setup/tests.log
touch runs/setup/ENVIRONMENT_COMPLETE
