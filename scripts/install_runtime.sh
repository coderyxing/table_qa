#!/usr/bin/env bash
# 脚本用途：等待轮子下载完毕后安装冻结的运行依赖并验证环境。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 在有限次数内轮询前置任务；退出条件和等待间隔由循环正文定义。
for attempt in $(seq 1 120); do
  if [[ -f runs/wheels/DOWNLOAD_COMPLETE && -f runs/cuda-wheels/DOWNLOAD_COMPLETE ]]; then break; fi
  # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
  sleep 10
done
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test -f runs/wheels/DOWNLOAD_COMPLETE
test -f runs/cuda-wheels/DOWNLOAD_COMPLETE
# 加载项目虚拟环境或阶段配置，使后续命令使用预期依赖。
source .venv/bin/activate
# 调用当前阶段的 Python 入口，参数中的文件明确给出本次输入与输出。
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
