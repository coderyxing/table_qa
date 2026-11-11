# GPU14 源码版本与 GitHub 同步

本次同步于 2026 年 11 月 11 日从 GPU14 取回源码，按文件内容校验后上传到 `coderyxing/table_qa`。

| 服务器目录 | 用途 | GitHub 中的对应版本 |
|---|---|---|
| `/memory/tableqa/runs/20261110-pilot/source/` | 首次部署的源码子集，共 65 个文件 | 文件内容全部包含在 source-v2 导入提交中 |
| `/memory/tableqa/runs/20261110-pilot/source-v2/` | 完整试跑版本，共 180 个文件。用于第三章首轮和第四章试跑 | 提交消息为 `Import complete GPU14 source-v2 experiment snapshot` 的版本 |
| `/memory/tableqa/runs/20261110-pilot/source-v3/` | 第三章题面引文定位修复版，共 180 个文件 | 后续同步提交的项目源码 |

source-v2 与 source-v3 的差异位于 `tableqa/chapter3.py`、`tableqa/thesis.py`、`tests/test_thesis_experiments.py`。source-v3 的文件与同步时的本地项目源码逐文件一致。各版本文件的 SHA-256 见同目录的 `server-source-manifest.json`。

服务器实际使用的两个启动脚本保存为 `scripts/server/launch-pilot.sh` 和 `scripts/server/continue-chapter3.sh`。脚本保留本次运行的绝对路径、固定模型提交、数据版本及独立源码目录。复现时先在目标环境确认这些路径、依赖及 GPU 可用情况，并按各自版本恢复源码快照。

运行产物位于服务器的 `/memory/tableqa/runs/20261110-pilot/`。模型位于 `/memory/tableqa/models/`，DataBench 文件位于 `/memory/tableqa/data/chapter4/`。模型、数据、日志和凭据通过项目忽略规则管理。

本次同步采用一次性核验和提交。后续修改源码后，需重新提交并同步 GitHub。
