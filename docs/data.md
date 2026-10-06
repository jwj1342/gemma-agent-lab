# Data description 与数据使用

官方全文已存为 [data-description.md](../resource/competition/data-description.md)。
本页结合本地完整下载审计解释如何使用，不覆盖原始快照。

## 文件清单

| 目录/文件 | 本地数量 | 用途 |
| --- | ---: | --- |
| `tasks.jsonl` | 129 行 | issue、仓库版本、公开参考修复与测试 |
| `snapshots/` | 129 TGZ | 去除未来修复历史的 Git 仓库快照 |
| `graphs/` | 127 JSON | 127 个唯一 repo/base_commit 的符号图 |
| `embeddings/` | 127 NPZ | 每个图节点 256 维 float32 向量 |
| `wheels/` | 124 wheels | 仓库测试依赖，不是 vLLM/ADK 运行依赖 |
| `docker/` | 4 文件 | Python 3.13 沙箱规格与兼容 shim |
| `sandbox/setup.py` | 1 文件 | 仓库依赖和测试环境准备；只在沙箱执行 |
| `sample_submission/` | 10 文件 | 官方 YAML、prompts、两个示例 LoRA |
| `HARNESS_README.md` | 1 文件 | 官方完整技术手册 |

总计 524 文件，20.88 GiB；ZIP 20.42 GiB。全部仓库展开还需约 21.13 GiB、23.8 万条目，
应按需展开。原文说 graph/embedding 各 256 文件，是包含 task-name hardlinks 的旧打包
口径；当前下载各 127 个 commit-name 文件，129 题全部能匹配，并无缺失。

## tasks.jsonl 字段

| 字段 | 含义 | 使用边界 |
| --- | --- | --- |
| `instance_id` | 题目唯一标识，如 `rich_3454` | 输出/trace/补丁以此关联 |
| `repo` | `owner/repo` | 构造路径时不能把 `/` 直接当普通文件名 |
| `base_commit` | 修复前 commit | 切分、快照、graph/embedding 配对依据 |
| `problem_statement` | issue 正文 | 解题输入 |
| `hints_text` | 维护者提示，可能为空 | 按 harness 协议提供给 agent |
| `patch` | 参考修复 diff | 训练标签或 gold 验证；禁止用于评估解题输入 |
| `test_patch` | 验证测试 diff | 隔离评分依据；正式评分中保密 |
| `created_at` | 题目时间 | 分布与切分分析 |

公开分布：FastAPI 67、Rich 48、Requests 13、HTTPX 1。按 repo/base_commit 分组建立
训练/验证切分，不能仅按行随机分后让同一个仓库快照跨两侧。

## Graph 和 embedding

图是 directed multigraph，节点含 `id`、`name`、`text`；边有 `source`、`target`、`type`、
`key`。本地 127 图的边类型实际只有 `calls`。文档中示例的其他关系名称不表示数据里
一定存在。NPZ 每个 `<node_id>.npy` 是长度 256 的 float32 数组。

先通过 issue/源码搜索得到真实符号，再查邻居、子图及相似符号。图中 async 函数缺失等
问题已被讨论；查不到节点不代表代码不存在。`search_similar_code` 不能未经验证就当成
任意自然语言向量搜索服务。图不是 gold patch，也不能保证包含所有真实依赖关系。

## 两个不同的 wheelhouse

`data/raw/.../wheels/` 支持旧仓库测试。独立 Kaggle 数据集
`metric/gemma-4-developer-agent-wheelhouse` 提供 vLLM、ADK、swegemma 等 harness 包。
后者的部分二进制 wheel 绑定 Python 3.12；Docker 沙箱使用 Python 3.13，二者不矛盾，
不能把一个环境的 wheel 无差别装进另一个环境。

公开仓库测试 wheels 缺少部分题需要的依赖。先验证一个选定题的 baseline 失败、gold
通过，再将该题用于训练 reward。缺包、collection failure、超时不是“模型修复失败”
的同义词。保留完整测试日志，勿为得到分数而修改验证测试。

## 输出与审计

我们提交的是 `submission.zip`。Data description 中的 `submission.parquet` 是评分链路
生成的中间输出，不是本团队当前要上传的公开题答案表。

完整记录：[文件清单](../resource/manifests/competition-files.json)、
[ZIP 校验](../resource/manifests/download-verification.json)、
[数据结构审计](../resource/manifests/data-audit.json)、
[示例 adapter 头](../resource/manifests/sample-adapters.json)。
