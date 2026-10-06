# Gemma 4 Developer Agent

团队私有仓库：[jwj1342/kaggle_gemma4_agent](https://github.com/jwj1342/kaggle_gemma4_agent)，默认分支 `main`。
维护者的 Trillium 工作区为 `/scratch/jwj/kaggle_gemma4_agent`；队友可在自己的电脑克隆，
参与开发不需要 Trillium 账号或 H100。
比赛：[Google — The Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)。

新成员先读 [团队上手与协作 / Contributing](CONTRIBUTING.md)、
[英文组会 slides PDF](slides/gemma4-kickoff/main.pdf)，再读 [Overview](docs/overview.md)、[Data 导读](docs/data.md)、
[参与指南](docs/participation.md)，再按 [完整资源索引](resource/README.md) 查原文。
首次组会的 32 页英文材料与构建说明见 [slides](slides/gemma4-kickoff/README.md)，已同步到
[Overleaf](https://www.overleaf.com/project/6ac31a4a617496d875323ad8)。技术判断见
[比赛调研](docs/competition-research.md)、[提交步骤](docs/submission.md) 和
[训练与评测准备](docs/training-plan.md)。资料快照采集于 2026-10-04 晚（UTC 为 10-05），
并不保证未来规则、排行榜或软件版本不变。

本次快照的 112 个公开 Code 条目已全部建立[逐项阅读记录](resource/notebooks/reading-notes.md)：
记录关键执行路径、可借鉴点和结论边界；其中 1 个条目源码为空，无法检查实现。
这些是源码阅读与静态核对，不代表复现了全部 notebook 或其声称的成绩。

| 目录 | 用途 |
| --- | --- |
| `src/` | 团队实现的 agent 工具、数据转换、训练和评测代码 |
| `configs/` | 可复现的环境、agent、训练和评测配置 |
| `scripts/` | 数据下载、校验和资料采集脚本 |
| `data/raw/` | 原始比赛包，已下载并验证 524 个文件 |
| `data/processed/` | 清洗数据和训练/验证切分 |
| `model/base/` | 指定 Gemma 底座 v2；已通过单题离线推理 |
| `model/adapters/` | 导出的 LoRA adapters；目前尚未训练 |
| `resource/` | 官方页面、公开 notebook、讨论、模型卡、榜单与来源清单 |
| `runs/` | 每次实验的日志、缓存、检查点；不进 Git |
| `docs/` | 团队维护的说明、判断和后续计划 |
| `slides/` | 首次组会英文 slides、可编辑 LaTeX 与校徽来源 |

## 克隆与第一次轻量检查

获得仓库访问权限并登录 GitHub 后：

```bash
gh repo clone jwj1342/kaggle_gemma4_agent
cd kaggle_gemma4_agent
# Python 3.10+；只使用标准库，不需要 GPU、模型或 Kaggle token。
python3 scripts/build_submission.py \
  --agent-dir configs/baseline \
  --output runs/local-packaging/submission.zip
```

这一步检查基线能否打包；真实推理从 [Kaggle 官方 starter](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)
开始，具体流程见 [参与指南](docs/participation.md)。新的配置在功能分支修改，经 PR 合入 `main`。

Git 中包含代码、配置、资料快照、轻量结果清单和 slides PDF。**数据、权重、下载包、
运行目录和凭据不会随 clone 获取**。清单内的文件路径、作业 ID 和账号额度是维护者的
历史运行记录；队友需使用自己的数据位置、环境和账号。第三方资料的来源与许可见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 已验证的工作区与运行结果

维护者本地原始文件位于 `data/raw/gemma-4-developer-agent/`：22,416,766,904 字节（20.88 GiB）。
ZIP 保留在 `resource/downloads/`：21,929,289,087 字节（20.42 GiB）。
两份合计约 41.30 GiB；无需现在展开全部 129 个仓库。

Kaggle CLI 使用 HOME 中的独立环境，凭据由客户端自动读取 `~/.kaggle/access_token`。
以下命令为维护者的 **Trillium 配置**，不作为其他电脑的通用安装命令：

```bash
bash -l scripts/kaggle.sh competitions files -c gemma-4-developer-agent
```

重新下载/检查归档应在 datamover 上运行（脚本使用当前账户路径）：

```bash
ssh -o BatchMode=yes trig-dm1 'bash -l /scratch/jwj/kaggle_gemma4_agent/scripts/download_competition.sh'
```

下载校验记录在 `resource/manifests/download-verification.json`。
无 LoRA 基线已经打包；私有 Kaggle CPU notebook v1 运行完成，下载的 ZIP 与本地一致。
记录见 `resource/manifests/kaggle-pipeline.json`。这验证的是打包和 notebook 输出链路，
完整官方编译、GPU 离线推理和独立仓库测试也已通过：公开题 `rich_3454` 的原始代码
1 项失败，gold 补丁 83 项通过，Agent 补丁 84 项通过（含新增的一项测试）。
作业 `1033882` 正常结束，用时 5 分 49 秒；证据见
[单题验证记录](resource/manifests/hpc-smoke.json)。这不是全量 benchmark 或训练结果。

同一份 ZIP 的正式提交 **56844353 已完成，公开分数 0.12（12%）**；
2026-10-05 18:14 UTC 查询无评分错误，private 分数尚未提供。
私有 **L4×4 验证也已完成**：`rich_3454` 的 empty/gold 对照有效，Agent 补丁 84 项
测试通过，Agent 阶段约 74.10 秒（不含模型启动），notebook 到 smoke 完成约 655 秒。
这是单题验证，不是全量 benchmark 或训练结果。
结果见 [正式评分](resource/manifests/competition-result.json)、
[L4 验证](resource/manifests/kaggle-runtime-smoke.json) 和 [提交指南](docs/submission.md)。

协作时每个改动附简短目的、使用的配置及验证结果。不要把公开 notebook 的标题成绩
当作可复现结论。Git 仅负责源码历史；SCRATCH 无备份，重要源码/结果还应按团队约定
备份到 PROJECT 或独立的结果存储。GitHub 邀请和 Kaggle 正式组队分别管理；仓库创建
不会自动授予队友权限。当前正式训练方案仍在讨论，尚无已完成的微调结果。
