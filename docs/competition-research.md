# 比赛调研与操作约束

采集开始于 2026-10-04 晚，API 文件时间为 2026-10-05 UTC。
以下区分官方规则、主办方论坛澄清、社区报告及本地静态核查。
**Overview 与 Data description 均已完整取得**，保存在 `resource/competition/`。

## 任务、评分和提交

- 输入是软件仓库快照和 issue；agent 在离线沙箱修改源码，提交补丁，验证环境应用
  补丁并运行测试。得分是成功解决任务的比例；不是 CSV 分类预测。
- 提交 `submission.zip`，根目录使用 `agent.yaml`。工具由 harness 注册；自定义 Python
  代码作为 skill 的脚本在沙箱内运行，不能提供任意宿主 Python 入口。
- 允许 prompts、ADK 配置、skills 和可选 PEFT LoRA。展开后总大小保守控制为 **<3 GiB**，
  包括 adapter；支持的扩展为 `.yaml/.yml/.md/.txt/.py/.json/.safetensors`。
  不带 symlink、越界路径、pickle 权重或其他压缩包。
- 当前所有 agent/subagent 都必须使用 **`gemma-4-31b-it-qat-w4a16-ct`**。
  小模型未来支持的论坛计划尚不能视为当前许可。
- 每天最多 **1 次提交**、最终最多选 **2 个**、团队最多 **5 人**。
- 约 **120 个隐藏任务**，public/private 各约一半，同一次 12 小时执行内生成补丁。
  隐藏仓库来自**非公开仓库**，不是公开四个仓库的随机保留题。
- 12 小时包含 sandbox setup，排除补丁验证；每任务 agent 时间预算不含 setup，二者
  不能混同。主办方确认任务按顺序处理。按 120 题平均最多约 6 分钟，还需扣除 setup
  和其他开销，因此初始 cap 可用 4–4.5 分钟作待测方案，不能保证在 L4 上一定按时完成。
- 9 月 28 日主办方仍说明全局超时会报错；尚未找到“未完成自动记零”已上线的明确确认。
  不把计划中的修复当作可依赖行为。
- 赛程：11 月 12 日可选论文截止；11 月 25 日报名/合队截止；12 月 2 日最终截止，
  均以官方所列 23:59 UTC 为准。预测赛奖金 $65,000，另有 $35,000 paper track。

来源：[Overview](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)、
[Rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules)、
[任务数澄清](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744258)、
[隐藏仓库](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744951)、
[顺序及预算](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743063)、
[超时与并行澄清](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743964)。

## 官方计算资源：L4×4

是 **4×NVIDIA L4，每卡 24 GB，共 96 GB VRAM**。比赛关联的 Kaggle Notebook 可选择
该 accelerator，须关闭 Internet。通常先从官方 starter 点 Copy & Edit，确认 competition
data、官方 wheelhouse 和指定模型已作为 Inputs 附加，在 Session options 中选 L4×4。
保存运行的 notebook 必须实际输出 `/kaggle/working/submission.zip`，随后选择它提交。
未实际打开此账号的网页 UI；按钮名称以当时页面为准。

- L4×4 按旧 T4×2/P100 的 **2 倍速率**消耗 GPU quota，官方保留调整倍率的权利。
- 2026-10-05 原始 API 确认本账号 GPU quota 为 **108,000 秒（30 小时）**，发起本次
  L4 smoke 前已用/预留 0。按当前 2 倍倍率，折合约 15 小时 L4×4 开发运行额度；
  这不是单个 notebook 可以连续运行 15 小时的保证。TPU 为 72,000 秒（20 小时）。
- **更正早期 6 小时记录**：kagglesdk 0.1.37 的 `TimeDeltaSerializer.serialize` 使用
  `timedelta.seconds` 而非总秒数，丢掉一天，造成 30 小时被转存为 21,600 秒。
  早期 `accelerator-quota.json` 保留供追溯，但不能作为额度依据；不是账号配额刚增加。
  今后读取原始响应或 `total_seconds()`，并同时记录已用及预留额度。
- 刷新时间为 `2026-10-10T00:00:00Z`。执行前重新查看 quota，资源还可能排队。
- 比赛评分的 12 小时上限与 Notebook 个人额度是不同限制。
- 本地 Trillium H100 更快、更大，并不保证同一个 time budget 在 L4 上有相同结果；
  最终需做接近评分环境的时间与显存验证。

来源：[Upgraded Accelerators 原文](../resource/competition/upgraded-accelerators.md)、
[Google Cloud L4 规格](https://docs.cloud.google.com/compute/docs/gpus)、
[账号 quota 原始响应](../resource/competition/accelerator-quota-raw.json)。

## Data：已逐项清点的内容

完整数据包 524 文件，22,416,766,904 字节（20.88 GiB）；ZIP 21,929,289,087 字节。
全部文件与官方 API 名称/大小一致，解压时校验 ZIP CRC。
另对所有任务、图、NPZ 头、wheel metadata 和 TGZ 成员做结构检查，详见
[审计清单](../resource/manifests/data-audit.json)。没有执行仓库代码或运行测试。

| 内容 | 数量 | 实际用途与发现 |
| --- | ---: | --- |
| `tasks.jsonl` | 129 题 | FastAPI 67、Rich 48、Requests 13、HTTPX 1；含 issue、hints、base commit、参考 patch 与 test patch |
| `snapshots/` | 129 TGZ | 仓库快照；全部成员合计 237,734，全部展开约 21.13 GiB；按需展开 |
| `graphs/` | 127 JSON | 对应 127 个唯一 repo/commit；711–4597 个节点/图，边类型实际均为 `calls` |
| `embeddings/` | 127 NPZ | 每符号 256 维 float32；已遍历数组头，不进行不必要的 pickle 反序列化 |
| `wheels/` | 124 wheels | 41 个不同包的多版本仓库测试依赖；不是模型服务/训练 wheelhouse |
| `docker/` | 4 文件 | 两份 Python 3.13 Dockerfile、imp/telnetlib 兼容文件 |
| `sandbox/setup.py` | 1 | editable install、依赖选择、工作目录注入与测试配置；不能直接在宿主运行 |
| `sample_submission/` | 10 | analyzer + coder 的 YAML、prompts、sampling、预算、两个示例 adapter |
| `HARNESS_README.md` | 1 | 671 行技术说明，已阅读提交、工具、预算、两阶段评测和本地 CLI 部分 |

所有 129 题均找到对应 snapshot、graph 和 embedding。Data description 的“256 个图/
256 个 embedding，含 task 名 hardlink”是旧打包说明；当前包只有各 127 个 commit 名文件。
不能把这个差异当作下载丢失。字段映射保存在 `tasks-audit.json`。
两个示例 safetensors 的头部也已读取，各有 4 个张量，文件内 offset 范围检查通过；
[头部记录](../resource/manifests/sample-adapters.json) 保留形状、dtype 与 hash。
这仅验证文件结构，不代表它们训练过或有有效收益。

当前仓库测试 wheels 缺少 `typing-inspection`、`inline-snapshot`、`dirty-equals`、
`python-multipart`。社区也报告相关 collection error，但这不证明私有评分环境同样缺包。
先对选定公开题验证“基线失败、gold patch 通过”，再将它用于训练 reward 或评估。
不要把参考 patch/test_patch 输入给解题 agent；它们是训练标签或隔离的验证依据。

额外的官方数据集 [Gemma 4 Developer Agent Wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)
本次列出 41 个文件，包括定制 `vllm 0.19.1`、`swegemma 0.2.7`、
`adk-submission 0.2.12`、`adk-eval-core 0.1.0`、`google-adk 1.36.1`、
`transformers 5.13.1`。它与上面的 124 个测试 wheels 分工不同。
版本号不够：主办方可能同版本重新打补丁，应固定数据集版本和文件 hash。
调研阶段先下载三个小型 harness wheels，并提取源码作静态核对。后续已下载并校验完整
41 文件的官方 wheelhouse，另建立 native compiler 预检查环境和独立容器评测环境。
后者已通过完整 compiler、单题真实生成与独立测试，详见 [运行记录](cluster-runtime.md)。
源码证实 LoRA 槽数/rank 会按实际 adapter 推导，thinking budget/seed 有转发逻辑，
graph 工具已有输出截断代码。后两者不能仅凭论坛较早的 bug 报告判断仍未修复；
也不能把“源码已有处理”直接当作所有运行场景已验证。
下载 hash 见 [harness 清单](../resource/manifests/harness-wheels.json)。
此外，`adk_submission 0.2.12` 中 `include_thoughts=False` 会设置
`enable_thinking=False`，并非只隐藏推理文本；设置前须理解这会改变生成行为。
这与 [社区报告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745059)
一致，源码位置为 `adk_submission/resolvers/generation.py`。

来源：[Data description](../resource/competition/data-description.md)、
[原始 harness 指南](../data/raw/gemma-4-developer-agent/HARNESS_README.md)、
[全部文件](../resource/manifests/competition-files.json)、
[校验记录](../resource/manifests/download-verification.json)。

## Code：112 个公开条目的归档与审阅

按 competition 过滤、按创建时间分页获得 100+12 个条目，全部源文件下载成功。
3 个匿名 pull 返回 401 的条目已通过认证接口读取；1 个公开条目源码为空。
保留原始源码、metadata、hash 和可读文本；逐条做结构/依赖/配置审阅，重点审读官方
评测链路、训练入口、agent 打包、图检索与数据质量方案。**没有逐个执行 notebook，
也不把静态扫描称为所有代码逐行审计或成绩复现。** 全部条目见 [Code 索引](../resource/notebooks/README.md)。

2026-10-05 已补齐全部 **112 项逐项阅读记录**（含 1 项空源码的不可审阅说明）。
[阅读表](../resource/notebooks/reading-notes.md) 记录实际入口、默认开关、最终打包内容、
可借鉴点、疑点和源码行号；JSON 另保存每份可读源码的 SHA256。
部分 notebook 的实验在打包流程之外，只有真正写进 Agent ZIP 的逻辑才会被正式评测执行。

有直接参考价值的路径：

- [官方 Getting Started](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)：
  离线 wheels → 拷贝 sample → vLLM → 两题 local eval → 打包。
  示例会全局强制重装 packages、删除 cutlass `.pth`，不能原样搬到 HPC 宿主。
- [真实本地 evaluator](https://www.kaggle.com/code/hitarthjain0/gemma-4-agent-run-the-real-evaluator-locally)：
  展示按任务 trace、重复调用与失败排查；适合设计本地实验记录。
- [0.13 提交示例](https://www.kaggle.com/code/hsiaosuan/gemma-developer-agent-0-13-submission)：
  顺序 read-only planner → coder，无 LoRA；5 分钟/60 calls/100 turns。分数为作者声明，
  不是我们重跑结果，也不能证明架构单独造成提升。
  已静态解码其嵌入 ZIP 并核对 YAML；两阶段都设置 `include_thoughts: false`，在本次
  harness 版本中会关闭 thinking，因此不能按 YAML 的非零 thinking budget 推断实际推理开启。
- GraphLoc、Wold 的 graph coverage/localization、code-graph-audit 等：揭示 async 节点、
  名称解析、图工具覆盖与定位问题；适合作为可测的检索 baseline，不把 gold 标签带进推理。
- [31B QLoRA study](https://www.kaggle.com/code/denpugovkin/can-gemma-4-fix-an-unseen-repo-31b-qlora-study)：
  有训练与导出代码，仍需核对底座/模块/版本、切分和本机兼容性。
- [TPU KerasHub 教程](https://www.kaggle.com/code/nursrijan/gemma-4-devagent-kerashub-lora-training-on-tpu)：
  训练失败分支生成随机示例张量、导出层名只是字符串替换；不能视为比赛 adapter 已兼容。
- [TFD-Agent 研究 notebook](https://www.kaggle.com/code/rajrajak99/tfd-agent-autonomous-swe-bench-resolution-with-ge)：
  展示架构与文件操作模拟；图中的 0.18 是写死的预估值，不是本 notebook 跑出来的验证成绩。

共同风险：旧 hardlink 修复代码可能已过时；使用 mock validator 通过不等于官方 compiler
通过；把采样预算设到 10/20 分钟可能超过总时限；打包成功不等于模型已运行；
“Top 1 / SOTA”标题不能替代当前榜单或完整实验记录。

进一步阅读后的优先方向与边界：

| 候选方向 | 已看到的实现依据 | 下一步如何验证 |
| --- | --- | --- |
| 有界源码定位 | Evidence Scout 的多视图 BM25/RRF、GraphLoc 的词法/图消融 | 固定 issue 与源码输入，对比 grep/词法 baseline；同时记录解题率和耗时 |
| 减少无效循环 | lean/reviewer/full 配置、重复调用统计、先失败再通过的反例探针 | 同预算、同任务比较；先确认探针复现问题，再判断修复，不按调用成功当测试通过 |
| 评测证据与环境控制 | Shape of Doubt 保存 ZIP hash、固定补丁重放、端点健康与 JUnit 记录 | 环境变更单独验证，保留失败；避免把依赖修复带来的通过归因于新提示词 |
| SFT / LoRA | 31B QLoRA study 的真实训练入口；部分脚本有 diff-only loss mask | 先证明底座兼容和小量训练恢复，再做不使用 gold 定位的跨仓库评估 |

这些都是待测假设。尤其注意以下源码中可确认的情况：

- 一些图实验使用 gold 文件全集作为候选目录，或直接把目标 embedding/符号作为 query/seed；
  这类指标不能解释为未知 issue 的定位效果。另一些脚本用新文件的 diff 行号对应旧快照 AST，
  标签需要重算。部分“重建图”类仅被定义，实际比较没有调用它。
- TPU 教程实际从 Gemma 2 2B 开始，导出时改写 Gemma 4 31B 名称；另有脚本生成随机 A、
  全零 B 或无效占位文件。它们不构成已兼容或已训练的比赛 LoRA。
- `RUN_HF_SMOKE=True`、`RUN_COMMAND_SCREEN=True` 等开关可能与“纯 CPU/默认不运行”的
  说明不一致。相反，有真实训练函数也不意味着默认运行过训练。
- `119 of 129 sound` 的验证统计来自作者历史审计表/附加 CSV，当前 notebook 主要重算统计
  与定位；其 raw required-node surrogate 是 117，119 含作者对 skip 的解释。不能当作
  我们已逐题复验、或当前官方私有环境的成功率上限。
- Black Cat v13 保留的 0.12 属于 v10 anchor；v13 的内嵌 L4 pilot 记录是服务启动超时、
  完成 0 对任务。默认选择新候选并不会继承旧包成绩。
- `complete EDA & ADK starter kit` 的末尾 parquet 直接填入参考 `patch`；这是答案导出，
  不是 Agent 预测。正式提交仍应使用经过官方编译和实际推理验证的 Agent ZIP。

以上逐条出处及原始 notebook 链接均在[阅读表](../resource/notebooks/reading-notes.md)，
没有执行第三方 notebook，也没有将这些候选加入当前正式 baseline。

## Models：本次页面的 4 个条目

直接读取网站的 `ListCompetitionModelInstances`，当前无下一页 token。

| 页面条目 | 版本/大小 | 判断 |
| --- | --- | --- |
| `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct` | v2，23,297,590,856 字节（约 21.70 GiB） | 当前正式底座；该尺寸是整个模型包，不能仅按 README 的 16–18 GB 理论权重估算存储 |
| `jek1wantaufik/buddy/Transformers/gemma4` | v1，8,989 字节 | ModelBuddy，说明实际指向 agent bundle；不是获准替代底座的证据 |
| `mousumiroy250/mt5-base/Transformers/default` | v1，约 6.99 GB | 页面附带条目；没有正文模型卡；不允许替代正式底座 |
| `muhammadukasha09/bert-base-multilingual-cased/Transformers/default` | v1，约 717 MB | 页面附带条目；没有正文模型卡；不允许替代正式底座 |

已保存四个模型的 metadata、实例说明和存在的模型卡，未批量下载权重。
Gemma 家族卡里的 128K/256K context 不覆盖比赛的 **32,768 token** 总上下文限制。
Models 页 `bestPublicScore` 只反映页面关联信息，不代表整个榜单最优分。
来源：[Models 页](https://www.kaggle.com/competitions/gemma-4-developer-agent/models)、
[原始页面清单](../resource/models/competition-tab.json)。

## Discussion：116 个主题、370 条首帖及回复

全部主题首帖及回复树均已保存，数量与 API 报告对应；保留作者与时间。
主题级审阅及主办方答复交叉核对见 [讨论索引](../resource/discussions/README.md)。
外链、截图中的文字、Discord 和被删除/私有内容不等于已完整读取。

重要官方澄清：

- [ADK 版本](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743800)：
  以比赛 wheelhouse 为准，不支持 ADK 2.x，评分环境不能被参赛者修改。
- [LoRA 修复](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743508)：
  主办方称 adapter 清零问题在 wheelhouse v23 修复。
- [10 月 1 日状态](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744794)：
  主办方称 LoRA KV cache 配额已按真实 adapter/rank 动态设置，double-JSON、thoughts 丢失、
  thinking_budget/seed 转发均在最新包修复。README 中固定 max_loras=8/max_rank=128
  的描述不应直接当成现时动态服务策略；仍要实测 adapter 开/关有输出差异。
- [read_file 类型修复](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744678)：
  主办方称当前 wheelhouse 已修复。
- [公开题质量](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743817)：
  主办方承认公开题没有和隐藏题同等程度的复审；隐藏 gold patch 全通过的保证不能
  用来断言我们的公开 Docker/subprocess 环境已正确。
- [图文件](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742911)：
  空 embedding 打包问题已修复，但 async 函数缺失当时尚未修复。
- [外部教师数据](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807)：
  主办方允许符合模型许可及赛事规则的外部模型输出；记录来源、许可和生成过程。
- [自定义工具](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743573)：
  应打包为 skill 脚本在沙箱执行，不是宿主 registry 任意注册。

仍需跟踪：12 小时超时恢复是否上线、compaction 丢工具结果、未声明工具导致中断、
缺失公开测试依赖、含糊 issue（如 fastapi_15661）、chat template 修订、L4 排队。
这些主题里有未获确认的社区观察，不能全部当作现行官方 bug。

## Leaderboard：完整分布快照

官方 CSV 时间 `2026-10-05T03:50:43Z`；1,755 队；最高 0.24，中位数 0.08，均值约 0.08199。
榜首为 `samson8` 所在队，2–7 名同为 0.17。分数仅保留 CSV 展示精度。

| 展示分数 | 队伍数 |
| --- | ---: |
| 0.24 | 1 |
| 0.17 | 6 |
| 0.15 | 41 |
| 0.13 | 139 |
| 0.12 | 307 |
| 0.10 | 344 |
| 0.08 | 316 |
| 0.06 | 200 |
| 0.05 | 148 |
| 0.03 | 65 |
| 0.01 | 55 |
| 0.00 | 133 |

大量并列，少数任务的成功率变化即可明显影响排名。社区从离散分数推测 public 有
58 题，这不是官方确认，不能把它当成准确隐藏题数。官方只确认约 120 总题数、约一半
public/private。当前公开排名也不能预测 private 最终成绩。
来源：[Leaderboard](https://www.kaggle.com/competitions/gemma-4-developer-agent/leaderboard)、
[统计 JSON](../resource/leaderboard/summary.json)。

## 当前成果边界

数据、主要文字页面和可枚举公开资源均已取得；没有 Overview/Data description 的读取阻塞。
二进制数据做完整清单和格式审查；不是逐行阅读所有快照源码。社区 notebook 做全量结构
审阅加重点代码检查；没有完成 112 份代码的逐行安全审计或运行复现。
外链项目及图片附件未全部抓取。2026-10-05 后续已下载并校验指定模型 v2 与 41 个官方
依赖文件，建立无 LoRA baseline 并验证本地 ZIP 与私有 Kaggle CPU notebook 输出一致。
评测容器、独立环境及 Git 工具已准备；作业 1033882 正常完成完整编译、GPU 离线推理、
empty/gold 对照和 Agent 补丁独立测试（84 passed）。这是公开题 `rich_3454` 的单题验证，
没有训练或全量本地 benchmark。2026-10-05 18:14 UTC 查询确认，正式提交 56844353
已完成，公开分数为 0.12，private 分数尚未提供。私有 L4×4 单题验证亦已完成：
empty/gold 对照成立，Agent 84 项测试通过；Agent 阶段约 74.10 秒，不含模型启动。
这是既有提交的返回结果，没有重复提交。证据见 `resource/manifests/competition-result.json`
与 `resource/manifests/kaggle-runtime-smoke.json`。
最新步骤见 `docs/submission.md` 与
`docs/cluster-runtime.md`，下载证据见 `resource/manifests/eval-assets-verification.json`。
