# 模块职责：将 python -m tableqa 转发到统一 CLI；命令参数由 cli.main 解析。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
from .cli import main
main()
