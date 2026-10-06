# 公开 Notebook 逐项阅读表

当前 112 个公开条目；已完成 112 项阅读记录。本次快照已全部逐项记录；其中空源码条目仅记录不可审阅的边界。
本表区分实际写入的提交内容、notebook 外围实验和作者自述。没有执行第三方 notebook，也不是逐行安全审计。
原始下载不改写；嵌入的文字配置可以静态解码，不能因此认为其中脚本已运行或模型已训练。

所有条目的原始入口见 [完整索引](README.md)，已有重点结论见 [比赛调研](../../docs/competition-research.md)。

## 1. Chronos: Completion-Authority-Separated Agent Harn

[Kaggle](https://www.kaggle.com/code/segamega/chronos-completion-authority-separated-agent-harn) · [本地源码](public/segamega/chronos-completion-authority-separated-agent-harn/readable.txt)

**类型：本地评测与日志审计。** 复制官方 sample，运行前五题 Evaluator；额外记录 claim/resolved、事件链和 divergence。最终 ZIP 只打包 Agent 目录。

可借鉴：保存真实补丁、测试 verdict 和异常；将“产生补丁”与“解决问题”分开统计。

边界与疑点：GATE 仅改变 notebook 内的 preds，没有写入最终 Agent ZIP；hash-chain 审计模块也不在提交包中。不能把外围实验直接当作评分服务会执行的策略。

阅读定位：L180–L230, L260–L349。

## 2. Agentic | Counterexample Lab

[Kaggle](https://www.kaggle.com/code/lucifer19/agentic-counterexample-lab) · [本地源码](public/lucifer19/agentic-counterexample-lab/readable.txt)

**类型：反例探针 skill 的提交构建。** 构建单 coder 和 probe.py：最多三项探针，先保存 assertion-fail 基线，再按相同 hash 重跑；区分 ERROR/TIMEOUT/PASS。

可借鉴：生成探针要先验证能复现，再验证修复；保留断言并限制探针时间，helper 失败时继续正常修复。

边界与疑点：notebook 主要做 CPU 打包，GPU pilot 与得分未在此独立复现。include_thoughts=false 在当前定制 harness 会关闭 thinking；冻结文件与 agent 同属一个沙箱，不是不可篡改的评分边界。

阅读定位：L1–L28, L41–L127。

## 3. Gemma 4 Agent: Run the Real Evaluator Locally

[Kaggle](https://www.kaggle.com/code/hitarthjain0/gemma-4-agent-run-the-real-evaluator-locally) · [本地源码](public/hitarthjain0/gemma-4-agent-run-the-real-evaluator-locally/readable.txt)

**类型：真实 evaluator 多题入口。** 按 tasks[::13][:10] 跑官方 Evaluator，统计 patch、测试状态、命令次数和重复命令。作者报告同配置多次结果波动。

可借鉴：复用逐题诊断和重复调用计数；同一配置重复运行，避免根据一两题差异做结论。

边界与疑点：10 分钟每题不能直接覆盖约 120 题的 12 小时总预算；样本不是按仓库留出。重复检测把命令截为前 200 字符，可能把不同长命令合并；3/10 等为作者报告。

阅读定位：L1–L22, L87–L94, L228–L243, L250–L310。

## 4. Gemma Developer Agent: 0.13 Submission

[Kaggle](https://www.kaggle.com/code/hsiaosuan/gemma-developer-agent-0-13-submission) · [本地源码](public/hsiaosuan/gemma-developer-agent-0-13-submission/readable.txt)

**类型：已打包配置发布。** base64 ZIP 内是 SequentialAgent：只读 explore 先输出 repair_plan，coder 再编辑和提交；无 LoRA，5 分钟/60 tools/100 turns。

可借鉴：短规划交接、限制图查询、固定提交字节与 SHA256，适合作为受控对照。

边界与疑点：0.13 为作者自述，未独立复现。已静态解码内嵌 YAML；include_thoughts=false，不能按非零 thinking_budget 认定开启 thinking。

阅读定位：L1–L26。

## 5. Gemma 4 Developer Agent SWE Architecture

[Kaggle](https://www.kaggle.com/code/aryangupta85/gemma-4-developer-agent-swe-architecture) · [本地源码](public/aryangupta85/gemma-4-developer-agent-swe-architecture/readable.txt)

**类型：空源码条目。** 下载记录成功，但公开源码内容为空。

可借鉴：保留条目，避免把空文件误认为抓取遗漏或实现证据。

边界与疑点：无法审阅其架构、执行路径或结果；不能从标题补造实现。

阅读定位：下载源码为空。

## 6. The Gemma 4 Developer Agent Complete EDA

[Kaggle](https://www.kaggle.com/code/abelmatas/the-gemma-4-developer-agent-complete-eda) · [本地源码](public/abelmatas/the-gemma-4-developer-agent-complete-eda/readable.txt)

**类型：数据 EDA。** 扫描文件清单、任务/schema、diff 规模、graph/NPZ/snapshot/wheel 与 sample 配置；用正则给任务附启发式类别。

可借鉴：团队入门和数据分布分析；字段、体量及标签来源透明。

边界与疑点：类别特征混合 issue、gold patch、test_patch，是离线标签分析，不能作为无答案评测输入；token 数按字符估算，不是实测 tokenizer。没有 Agent 运行或得分证据。

阅读定位：L363–L374, L475–L515, L564–L650。

## 7. gemma4-agent-baseline-v2

[Kaggle](https://www.kaggle.com/code/juna1013/gemma4-agent-baseline-v2) · [本地源码](public/juna1013/gemma4-agent-baseline-v2/readable.txt)

**类型：可选单 agent 基线与评测。** 写四个主要配置文件，4 分钟/60 tools/80 turns；可选固定 seed 抽五题执行真实 Evaluator，但 RUN_LOCAL_EVAL 默认 false。

可借鉴：显式预算、固定抽样、区分仅打包和开启 GPU 实验。

边界与疑点：编译检查用 dummy tool 签名，因此不等同官方工具完整行为验证；默认跑完 notebook 不能证明 GPU 评测通过。

阅读定位：L1–L31, L266–L300, L350–L419。

## 8. Gemma 4-V0 | Walkthrough & First Submission

[Kaggle](https://www.kaggle.com/code/liujiahua2358/gemma-4-v0-walkthrough-first-submission) · [本地源码](public/liujiahua2358/gemma-4-v0-walkthrough-first-submission/readable.txt)

**类型：版本对照导读与基线构建。** 解析 sample、示例 LoRA、graph/embedding，并构建 analyzer+coder 的迭代版本；末尾打包提交。

可借鉴：读取每次版本变更及其来源；检查 graph 输出大小、正确读取测试尾部和固定 wheel hash。

边界与疑点：同一文档含旧版与后续修订说法，thinking/seed 转发须以当前 wheel 源码为准；引用的榜单分数不是其独立消融证明。

阅读定位：L1–L15, L275–L318, L380–L410, L630–L653。

## 9. Gemma 4 Where Is the Bug - Hop Distance Analysis

[Kaggle](https://www.kaggle.com/code/hajimekumami/gemma-4-where-is-the-bug-hop-distance-analysis) · [本地源码](public/hajimekumami/gemma-4-where-is-the-bug-hop-distance-analysis/readable.txt)

**类型：图定位离线分析。** 内嵌 gzip/base64 的 analysis.py 做 gold edit-site 映射、issue anchors、最短 hop、检索及 leave-one-repo-out；另修补 commit 文件名定位。

可借鉴：按仓库留出选择定位超参；分别统计无 anchor、无 gold node 与图距离。

边界与疑点：已静态解码脚本文本，未执行。原实现使用仓库 clone/git show，迁移离线环境应改用已给 snapshots；涉及 gold 的标签统计不能直接喂给解题 Agent。

阅读定位：L26–L45。

## 10. gemma4-agent-run1

[Kaggle](https://www.kaggle.com/code/sujanmajhisuzan/gemma4-agent-run1) · [本地源码](public/sujanmajhisuzan/gemma4-agent-run1/readable.txt)

**类型：嵌入基线的 GPU 运行适配。** 解码已有 bundle，前八题真实 Evaluator；按 GPU capability 选 dtype，并为 T4 路径设置 disable-custom-all-reduce。

可借鉴：诊断模型启动、记录逐题运行结果和 bundle 一致性。

边界与疑点：跳过若干官方二进制 wheels 的安装，且专门考虑 T4；不能据此声称与正式 L4 环境等价。未独立复现其吞吐或正确率。

阅读定位：L17–L30, L114–L169, L202–L244。

## 11. Gemma4-Agent（0.12points）

[Kaggle](https://www.kaggle.com/code/liuuu114514/gemma4-agent-0-12points) · [本地源码](public/liuuu114514/gemma4-agent-0-12points/readable.txt)

**类型：EDA、词法定位与提示词构建。** TF-IDF 定位分析后写 coder/analyzer prompts 和 ZIP；提示词增加复现、编辑后编译、测试及有限重试。

可借鉴：短定位报告和失败分类可用作 prompt 消融；EDA 与真实评测分开。

边界与疑点：提示词要求在已修改工作树上再次 git apply --check 正向 diff，失败时 checkout，可能误判并撤销有效修改；不能原样照搬。缺数据时有 synthetic demo；标题 0.12 不是独立复现实测。

阅读定位：L187–L191, L1253–L1274, L1280–L1315。

## 12. Gemma 4 Agent Research Artifact

[Kaggle](https://www.kaggle.com/code/jek1wantaufik/gemma-4-agent-research-artifact) · [本地源码](public/jek1wantaufik/gemma-4-agent-research-artifact/readable.txt)

**类型：研究产物归档。** 读取 buddy 模型页附带的 experiment_card/submission/bundle，解析 includes、哈希文件并复制为 paper_artifact。

可借鉴：按文件记录来源、配置和 hash，便于论文与复现实验交付。

边界与疑点：输入是第三方已保存的产物；manifest、配置检查和复制不证明训练或评测执行，也不使 Models 页附带模型成为允许底座。

阅读定位：L18–L38, L535–L583, L765–L813。

## 13. gemma4-data-eda

[Kaggle](https://www.kaggle.com/code/cylik93/gemma4-data-eda) · [本地源码](public/cylik93/gemma4-data-eda/readable.txt)

**类型：检索与图覆盖 EDA。** 从公开 gold diff 定位修改行，测 grep 覆盖、文件长度、图缺失和工具输出截断，保存 per_task.csv。

可借鉴：评估“目标在检索结果中”与“目标在截断后实际可见”之间的差距。

边界与疑点：5,000 字符、150 行等工具限制应随当前 harness 核实；基于公开分布的 hints 为空不能当作所有隐藏题保证。Gold 只作标签，未执行完整 SWE Agent。

阅读定位：L64–L68, L92–L118, L202–L221, L469–L487。

## 14. Gemma 4 Developer Agent - Lean

[Kaggle](https://www.kaggle.com/code/jek1wantaufik/gemma-4-developer-agent-lean) · [本地源码](public/jek1wantaufik/gemma-4-developer-agent-lean/readable.txt)

**类型：已有模型页产物检查与重打包。** 从 buddy 输入选择 submission 或 submission_bundle，读取 experiment_card、YAML 和 ZIP SHA，再复制并重打包。

可借鉴：追踪配置来源、includes 与归档一致性。

边界与疑点：不存在新的训练/推理入口；依赖额外挂载的第三方产物。模型页条目名称不是许可模型校验。

阅读定位：L12–L38, L255–L278, L378–L425。

## 15. Gemma4 Prompt Improvement Agent

[Kaggle](https://www.kaggle.com/code/kevlarzanderchi/gemma4-prompt-improvement-agent) · [本地源码](public/kevlarzanderchi/gemma4-prompt-improvement-agent/readable.txt)

**类型：提示词变体与回退包。** 输出 v12、原 v10 和 v11 三个 ZIP；v12 在已有系统提示词中加入 inline AST helper 与环境说明，其余文件保持明确差异。

可借鉴：保留精确 anchor 与单项变更，使用文件 hash 确认比较对象。

边界与疑点：三个包均未带 eval_config，预算依赖评分默认；自写 validator 不代替官方 compile 和真实任务。非零 thinking_budget 与 include_thoughts=false 并存。

阅读定位：L1–L12, L15–L45, L69–L128。

## 16. Gemma + Gemini - The Ultimate Duo

[Kaggle](https://www.kaggle.com/code/matterhorn3838/gemma-gemini-the-ultimate-duo) · [本地源码](public/matterhorn3838/gemma-gemini-the-ultimate-duo/readable.txt)

**类型：多基线 profile 构建与 EDA。** 默认 top_bundle；custom 分支可切单 agent、图工具、提示词规则和采样设置；按 profile 打包并记录 diff/hash。

可借鉴：显式开关与已知 anchor 的字节比较，便于消融控制。

边界与疑点：标题含 Gemini，但检查的提交路径仍构建指定 Gemma 配置；不应据标题推断允许第二底座。将 include_thoughts=false 解释成仅隐藏文本不适用当前定制 harness；默认无 eval_config，作者引用分数不构成因果证据。

阅读定位：L80–L113, L1880–L1923, L2144–L2156。

## 17. pass2

[Kaggle](https://www.kaggle.com/code/abelog/pass2) · [本地源码](public/abelog/pass2/readable.txt)

**类型：图扩展研究原型。** 比较 centroid 检索、固定 hop 与自适应扩展，统计 patch/test_patch 文件召回和上下文大小。

可借鉴：分开衡量检索命中与上下文开销，再接真实 issue 查询和 SWE 测试。

边界与疑点：核心 query_vector=vectors.mean(axis=0)，没有使用 issue 文本；当前是基础设施诊断，不能称为 issue-to-code 语义定位或端到端解题率。

阅读定位：L1003–L1044, L1355–L1437。

## 18. gemma4-graph-coverage-notebook

[Kaggle](https://www.kaggle.com/code/fawad07/gemma4-graph-coverage-notebook) · [本地源码](public/fawad07/gemma4-graph-coverage-notebook/readable.txt)

**类型：论文结果 CPU 重放入口。** 从额外挂载的 repro 数据集复制脚本和重建图，重算覆盖/检索表；Agent 对比表读取已存 summary。

可借鉴：将可重算的 CPU 指标与引用的 GPU 运行结果分开标注。

边界与疑点：关键实现和缓存来自额外数据集；此 notebook 本身不重跑 L4 Agent。重放报告不等于独立产生同样的补丁与测试结果。

阅读定位：L1–L10, L25–L42, L65–L81。

## 19. repair-manifold-ablation

[Kaggle](https://www.kaggle.com/code/dalezhong/repair-manifold-ablation) · [本地源码](public/dalezhong/repair-manifold-ablation/readable.txt)

**类型：已知位置的修复提示词消融。** 比较自由生成和 operator-first，筛选少 hunk 任务，从 gold patch 取目标文件，将文件内容交给模型生成整文件替换。

可借鉴：可作为给定正确文件后的局部修复实验；保存模型身份、迭代次数和测试状态。

边界与疑点：使用 gold 文件路径，且先应用 test_patch；不是真实无答案定位的端到端 Agent。加载失败会回退 Qwen，不能作为正式指定 Gemma 的成绩；网络安装和整文件截断还需单独处理。

阅读定位：L117–L196, L260–L288, L300–L381。

## 20. PatchPilot Gemma 4

[Kaggle](https://www.kaggle.com/code/shorooghahmadi/patchpilot-gemma-4) · [本地源码](public/shorooghahmadi/patchpilot-gemma-4/readable.txt)

**类型：配置打包与可选 CLI 评测。** coder 调 analyzer，加 repo_navigation skill；写 YAML 和自定义检查器，再打 ZIP，另给 swegemma eval 可选入口。

可借鉴：提交结构、includes 检查和正确区分 ZIP 与评分中间 parquet。

边界与疑点：每题 max_time_minutes=30，若120题打满预算将远超12小时，必须实测和收紧；旧路径探测仅检查 /kaggle/input 一层。包验证没有独立证明真实评分通过。

阅读定位：L1–L22, L58–L87。

## 21. Gemma 4: A 16384 Output Cap Refuses Long Prompts

[Kaggle](https://www.kaggle.com/code/dariushafshar/gemma-4-a-16384-output-cap-refuses-long-prompts) · [本地源码](public/dariushafshar/gemma-4-a-16384-output-cap-refuses-long-prompts/readable.txt)

**类型：上下文预算与已采集 trace 分析。** 以嵌入的调用计数和作者记录，分析 32768 context 下 16384 输出上限、compaction 延迟与时间余量，导出预算表。

可借鉴：记录每次 prompt/output 的实际 token 与开销；把 compaction 时间也纳入预算。

边界与疑点：这是作者 trace 的再分析，不是在每次运行时重新执行模型。58题 public/116题总量是推断而非官方保证；标题风险须结合当前 vLLM 请求与版本验证。

阅读定位：L1–L43, L304–L354, L619–L649, L690–L729。

## 22. Code graph audit

[Kaggle](https://www.kaggle.com/code/akhilchinta1505/code-graph-audit) · [本地源码](public/akhilchinta1505/code-graph-audit/readable.txt)

**类型：源码图重建与覆盖审计。** AST builder 加 async/nested、导入解析及 contains 边，按 gold 编辑范围生成标签；对比已有调用记录并重放 harness graph lookup。

可借鉴：将图覆盖、真实调用召回、工具输出和检索收益分开比较；保留数据与源码许可。

边界与疑点：recorded calls 来自额外数据集，不是 notebook 现场生成；图/检索指标不等于最终 SWE resolved。重建图的运行开销和允许接入方式另需验证。

阅读定位：L1–L14, L53–L88, L294–L368, L661–L679, L778–L800。

## 23. GemmaLoc: edit-site labels and a Gemma 4 rerank

[Kaggle](https://www.kaggle.com/code/giuseppecastelluccio/gemmaloc-edit-site-labels-and-a-gemma-4-rerank) · [本地源码](public/giuseppecastelluccio/gemmaloc-edit-site-labels-and-a-gemma-4-rerank/readable.txt)

**类型：定位标签与缓存 LLM rerank 重放。** 从 gold diff 标注节点，对八种检索方案评分；默认按精确 prompt hash 读取 Gemma 缓存，也提供 LIVE_URL 重跑入口和外部 SWE-bench 子集。

可借鉴：标签对齐、缓存 miss 即失败、配对比较及跨仓库泛化检查。

边界与疑点：默认不是实时 Gemma 推理；需要额外挂载缓存/外部数据，且当前流程联网 clone，不能直接放进离线 L4 会话。标题 top-1 翻倍是定位指标，非修复成功率。

阅读定位：L1–L10, L542–L601, L823–L859, L1628–L1636。

## 24. Vibranium-Bench: Deterministic SWE Benchmarking fo

[Kaggle](https://www.kaggle.com/code/thurshwinc/vibranium-bench-deterministic-swe-benchmarking-fo) · [本地源码](public/thurshwinc/vibranium-bench-deterministic-swe-benchmarking-fo/readable.txt)

**类型：图裁剪与评测接口演示。** 提供 AST 图和预算遍历示例；context-reduction 表按固定公式生成，Agent 回答为固定 synthetic_patch 加 sleep。

可借鉴：可看数据结构和接口教学；正式实验需替换为真实图、模型及验证流程。

边界与疑点：示例测试只验证简单算术，耗时来自 sleep，token reduction 非实测；这些输出不能用作模型速度、显存或 SWE 解题效果证据。

阅读定位：L355–L411, L458–L526。

## 25. Dhia-Gemma4AA

[Kaggle](https://www.kaggle.com/code/dhiaalhemdani/dhia-gemma4aa) · [本地源码](public/dhiaalhemdani/dhia-gemma4aa/readable.txt)

**类型：官方 starter 的小幅派生。** 相对已归档官方 starter，改 sample 来源为作者数据集，并把 max_output_tokens 从16384改为8192；其余运行结构一致。

可借鉴：直接比较固定 starter 的单项设置变更。

边界与疑点：真正 Agent 内容来自另一个 sample-submission 数据集，不能只看 notebook 相似度认为提交相同；仍需检查数据集 hash 与运行结果。

阅读定位：L70–L88。

## 26. notebook46419a07d8

[Kaggle](https://www.kaggle.com/code/ayushmldixit1/notebook46419a07d8) · [本地源码](public/ayushmldixit1/notebook46419a07d8/readable.txt)

**类型：模型加载排障草稿。** 尝试 Transformers Gemma4 多模态 AutoModel、device_map 和 CPU offload，末尾还有 vllm serve 命令。

可借鉴：参考检查类名、processor、设备映射和显存状态的排障思路。

边界与疑点：源码含杀死当前 kernel 的 SIGKILL 单元及重新加载片段；不适合 Run All。没有完成的 Agent ZIP/官方任务验证路径，不能当作可复现提交基线。

阅读定位：L156–L182, L323–L368。

## 27. SurgiGraph-Paper-Experiments

[Kaggle](https://www.kaggle.com/code/yukailiu/surgigraph-paper-experiments) · [本地源码](public/yukailiu/surgigraph-paper-experiments/readable.txt)

**类型：论文图表与统计模型分析。** 包含数据摘要、Dirichlet 热点后验、嵌入的定位结果和 patch simplicity 对照，产物面向 paper track。

可借鉴：启发式假设可转为独立可检验消融，如最小修改与上下文定位。

边界与疑点：部分 token 开销由 cost model 估算；修复率/回归率表有 hazard-model derivation，并非本 notebook 的真实 Agent 测试。不能把模型推演的数值当成运行增益。

阅读定位：L1288–L1318, L1380–L1391, L1425–L1450, L1719–L1728。

## 28. notebook703cae6987

[Kaggle](https://www.kaggle.com/code/phmtin1306/notebook703cae6987) · [本地源码](public/phmtin1306/notebook703cae6987/readable.txt)

**类型：数据文件手工探查。** 定位比赛目录，查看某个 snapshot、graph/embedding 文件以及 wheel 的内部目录。

可借鉴：快速理解文件布局和归档类型。

边界与疑点：只是数据浏览，没有训练、Agent 配置或真实任务评测。任务命名示例与实际比赛不同，不能直接照示例名称定位。

阅读定位：L1–L25, L160–L194。

## 29. Gemma4 Candidate v7 Compile

[Kaggle](https://www.kaggle.com/code/eeshsaxena/gemma4-candidate-v7-compile) · [本地源码](public/eeshsaxena/gemma4-candidate-v7-compile/readable.txt)

**类型：候选包官方编译检查。** 解码内嵌 ZIP，安装 wheelhouse，构造模型 registry 和签名相符的 stub tools，调用官方 compile_submission。

可借鉴：把正式编译验证与纯路径/扩展检查区分；记录 ZIP SHA256。

边界与疑点：tools 为省略号函数、模型 endpoint 为本地占位端口，没有模型推理。解码配置无 eval_config，预算走默认；include_thoughts=false。

阅读定位：L5–L12, L17–L44。

## 30. Gemma4 Baseline Agent v2 Compile Fixed

[Kaggle](https://www.kaggle.com/code/eeshsaxena/gemma4-baseline-agent-v2-compile-fixed) · [本地源码](public/eeshsaxena/gemma4-baseline-agent-v2-compile-fixed/readable.txt)

**类型：v2 配置的编译修复。** 逐文件 base64 解码，官方 compiler 检查后打包；coder/analyzer 都有工具配置，5分钟/48tools/100turns。

可借鉴：带具体 tool 签名的配置编译预检查，帮助定位 include/schema 错误。

边界与疑点：不是推理复现；model endpoint 与工具函数均为占位。不能把 compile OK 当成解题成功，且脚本没有固定 ZIP 时间戳。

阅读定位：L4–L15, L19–L40。

## 31. Gemma4 Baseline Agent v2 Packaging

[Kaggle](https://www.kaggle.com/code/eeshsaxena/gemma4-baseline-agent-v2-packaging) · [本地源码](public/eeshsaxena/gemma4-baseline-agent-v2-packaging/readable.txt)

**类型：v2 配置纯打包。** 逐文件解码同类 coder/analyzer 配置，检查 agent.yaml 根路径并压缩；thinking_budget=0，5分钟/48tools/100turns。

可借鉴：最小 CPU 打包路径，适合隔离上传格式问题。

边界与疑点：没有官方 compiler、模型或测试调用；不要与 compile-fixed 版本混淆。目录 exist_ok 可能保留旧文件，需干净目录与确定性包审计。

阅读定位：L1–L19。

## 32. Google-Gemma4: Multi-Agent + Graph-RAG  Base

[Kaggle](https://www.kaggle.com/code/dedquoc/google-gemma4-multi-agent-graph-rag-base) · [本地源码](public/dedquoc/google-gemma4-multi-agent-graph-rag-base/readable.txt)

**类型：多轮配置草稿，末尾为含示例 adapter 的打包。** 同一 notebook 多次覆盖 submission.zip；最终块复制 sample 的 main_lora/tool_lora，写 root+analyzer、技能目录和配置后打包。

可借鉴：识别最后实际写出的 payload，而不是前面版本的架构描述。

边界与疑点：示例 adapter 复制不等于训练；最终 timeout_seconds=900 是单命令限制，不能当成每题15分钟。部分展示/检索有 mock fallback；技能目录的可用性和最终配置仍需官方编译实测。

阅读定位：L340–L376, L705–L825。

## 33. Gemma4 Baseline Agent

[Kaggle](https://www.kaggle.com/code/eeshsaxena/gemma4-baseline-agent) · [本地源码](public/eeshsaxena/gemma4-baseline-agent/readable.txt)

**类型：无 GPU 的双 agent 打包。** 写 coder、只读 analyzer、提示词和 eval_config；5 分钟、32 tools、80 turns，最后检查 ZIP 根目录和体积。

可借鉴：小型可读的配置起点；明确任务预算和提交工具。

边界与疑点：未调用官方 compiler 或 Evaluator；目录/体积检查不证明工具签名和运行兼容。thinking_budget=0 且 include_thoughts=false。

阅读定位：L11–L46, L48–L79。

## 34. notebook-agent.ipynb

[Kaggle](https://www.kaggle.com/code/harigov63/notebook-agent-ipynb) · [本地源码](public/harigov63/notebook-agent-ipynb/readable.txt)

**类型：模拟教学代码及外部数据集上传代码。** 查找 test.csv，缺失时构造 dummy 行，写固定 mock_fix 到 submission.csv；另一函数通过打印和 sleep 模拟推理。

可借鉴：作为反例检查实际输入 schema、输出格式及是否真正调用模型。

边界与疑点：本比赛要求 Agent ZIP，CSV 和固定补丁不能替代。末尾含 dataset_upload，不能直接 Run All；模型名称也不同于规定底座。

阅读定位：L30–L72, L84–L104, L116–L146。

## 35. graphforge-digest

[Kaggle](https://www.kaggle.com/code/vinay0005/graphforge-digest) · [本地源码](public/vinay0005/graphforge-digest/readable.txt)

**类型：图提取兼容实现与定位基准。** 实现 GraphForge Lite，并以 issue 为查询比较 BM25、hash、BGE、RRF 和图传播；gold diff 用于定位指标。

可借鉴：比较有/无图传播及 lexical/embedding 检索，记录跳过原因。

边界与疑点：跳过没有 gold 节点及小于30节点的图，报告仅针对可纳入子集；兼容模式不补 async。ALLOW_PIP=true，BGE 下载需预备；定位命中率不等于修复成功率。

阅读定位：L1–L16, L1103–L1135, L1400–L1485。

## 36. Gemma 4 SWE: Task Map & 12-Hour Budget

[Kaggle](https://www.kaggle.com/code/officialabhinavsingh/gemma-4-swe-task-map-12-hour-budget) · [本地源码](public/officialabhinavsingh/gemma-4-swe-task-map-12-hour-budget/readable.txt)

**类型：任务分布与总时间预算计算。** 统计任务参考补丁规模，以720分钟总上限、90分钟准备预留和多档单题上限计算场景。

可借鉴：将准备开销与逐题开销一起预算；单题上限不是目标耗时。

边界与疑点：evaluation_tasks=len(df) 明确仅为公开集大小的情景演算，不是隐藏题数；补丁大小/issue长度不能直接预测可解性。

阅读定位：L75–L118。

## 37. Gemma 4 LB: 58 Tasks, Bronze Is a 124-Team Tie

[Kaggle](https://www.kaggle.com/code/dariushafshar/gemma-4-lb-58-tasks-bronze-is-a-124-team-tie) · [本地源码](public/dariushafshar/gemma-4-lb-58-tasks-bronze-is-a-124-team-tie/readable.txt)

**类型：历史榜单量化与私榜模拟。** 由两位小数分数推断公开题数，拟合经验贝叶斯先验，模拟不同相关性下的私榜和并列名次。

可借鉴：理解分数量化、同分和选择最好一次提交的偏差；保留假设与敏感性。

边界与疑点：58题是推断，未获官方确认；使用第6天863队旧快照，不能代替当前榜单。私榜结果从未观测，任务独立性、等分规模和相关系数均为建模假设。

阅读定位：L697–L723, L862–L881。

## 38. Hidden Interfaces – audit and execution.

[Kaggle](https://www.kaggle.com/code/murtadhaabdabbas/hidden-interfaces-audit-and-execution) · [本地源码](public/murtadhaabdabbas/hidden-interfaces-audit-and-execution/readable.txt)

**类型：隐藏接口与评测健全性审计。** 重建历史工作树，比较空补丁、gold补丁及重命名新增接口后的变体，记录导入路径、依赖、测试及失败类型。

可借鉴：控制实验区分程序修复能力与评测环境问题；确认测试导入的是工作树。

边界与疑点：merge时间/PR号重建需与真实快照比对；online模式会联网安装依赖。该 notebook 没有运行修复模型；gold及其变体只用于审计，不能作为正式预测。

阅读定位：L242–L271, L337–L358, L434–L466。

## 39. Gemma 4: Measure Before You Tune

[Kaggle](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-measure-before-you-tune) · [本地源码](public/dmitriigluzdov/gemma-4-measure-before-you-tune/readable.txt)

**类型：已有实验产物复核与 LoRA 决策记录。** 读取并校验外部 CSV/manifest，展示114/129 sound任务、按仓库留出、prompt渲染核对和 LoRA NO-GO；最后构建现有公开基线。

可借鉴：训练前先测加载效果、渲染一致性和修复成功率；损失下降需与未解决轨迹对照。

边界与疑点：图表读取历史结果，没有在本 notebook 重跑训练；936次是token长度核对，仅5次核对token身份。作者报告adapter 2/32低于base 4/32，不能将NLL下降当提分。健全任务数依赖其历史环境。

阅读定位：L24–L66, L107–L118, L165–L171, L220–L243。

## 40. GTUA - Stability-Augmented Graph-Tool Utility Audi

[Kaggle](https://www.kaggle.com/code/norikazuhaji/gtua-stability-augmented-graph-tool-utility-audi) · [本地源码](public/norikazuhaji/gtua-stability-augmented-graph-tool-utility-audi/readable.txt)

**类型：图工具效用的超参数稳定性审计。** 以 TF-IDF 和单跳图传播比较30种参数组合，用 gold 文件定位指标区分 Help/Hurt/Neutral，再按80%一致性分类。

可借鉴：检查图传播收益是否对参数和仓库敏感；单个最优设置不足以支撑一般结论。

边界与疑点：定位是代理指标，未跑修复Agent；80%阈值和增益阈值为方法选择，不能当比赛官方判断标准。

阅读定位：L209–L245, L265–L310。

## 41. Gemma 4 Triad LB score 0.03

[Kaggle](https://www.kaggle.com/code/kozykappa/gemma-4-triad-lb-score-0-03) · [本地源码](public/kozykappa/gemma-4-triad-lb-score-0-03/readable.txt)

**类型：coder/localizer/critic 三 agent 配置打包。** root可调用定位和补丁审阅工具agent；三个配置都关闭thinking，8分钟/100工具/140轮，附自定义静态验证及固定ZIP元数据。

可借鉴：用一次定位与一次审阅限制额外调用；确定性打包便于版本比较。

边界与疑点：标题0.03为作者自述；自定义验证不是官方运行证明。8分钟乘约120题已达16小时，需另测全局预算；只读职责是提示词约束，run_command本身仍可写。

阅读定位：L318–L395, L396–L420, L503–L535。

## 42. Google-TheGemma4DeveloperAgentCompetition

[Kaggle](https://www.kaggle.com/code/rifat2022/google-thegemma4developeragentcompetition) · [本地源码](public/rifat2022/google-thegemma4developeragentcompetition/readable.txt)

**类型：历史 code graph 缺陷修复和检索审计工具。** 包含文件别名/空文件修复、源码符号枚举（含async）、图重建及检索覆盖报告。

可借鉴：区分文件发布缺失与图结构遗漏，先做 census 再决定是否修复。

边界与疑点：开头256文件/硬链接丢失描述针对旧发布；我们当前快照127组图和embedding已核验非空并映射全部129题，不能照搬旧结论或盲目改原数据。定位/覆盖测试不证明Agent修复增益。

阅读定位：L1–L24, L355–L400, L506–L542, L984–L1015。

## 43. Eval errors - Gemma 4 Developer Agent

[Kaggle](https://www.kaggle.com/code/bharat0/eval-errors-gemma-4-developer-agent) · [本地源码](public/bharat0/eval-errors-gemma-4-developer-agent/readable.txt)

**类型：用 gold 补丁隔离测试评测阶段。** 注释模型服务启动，并替换 Evaluator._run_agent_sandbox，直接返回 task.patch 和空 trace，用于诊断评测第二阶段。

可借鉴：检查参考答案能否在指定环境过测试，定位安装或验证器错误。

边界与疑点：未运行Agent；gold通过率不是模型成绩。这段 monkeypatch 只能放独立控制实验，不能进入正式预测链路。

阅读定位：L161–L194, L218–L240。

## 44. Gemma 4 Dev Agent: task anatomy + survival kit

[Kaggle](https://www.kaggle.com/code/osama133/gemma-4-dev-agent-task-anatomy-survival-kit) · [本地源码](public/osama133/gemma-4-dev-agent-task-anatomy-survival-kit/readable.txt)

**类型：公开任务解剖和入门操作指南。** 从 gold diff 估计源文件数、hunk、改动行、函数和async特征，结合issue线索作图；附路径发现和文件导出辅助。

可借鉴：首次团队介绍、理解改动范围，并强调同一题集对照和重复运行。

边界与疑点：函数归因仅按diff hunk上下文，可能误归属或漏计；gold只作离线分析。无模型推理与比赛成绩证据，不应从补丁小推断任务容易。

阅读定位：L156–L203, L440–L449, L455–L489。

## 45. Gemma 4 Baseline Reproduction

[Kaggle](https://www.kaggle.com/code/ahmadagbaria/gemma-4-baseline-reproduction) · [本地源码](public/ahmadagbaria/gemma-4-baseline-reproduction/readable.txt)

**类型：EDA、词法定位与可选双 agent 打包。** 解析数据和参考diff，做TF-IDF定位；构造单agent或analyzer+coder，定制validator、确定性ZIP及experiment_card。

可借鉴：源码/配置/hash一并留档；分离定位指标与最终提交内容。

边界与疑点：源码与romanrozen条目逐字节相同，并不是第二次独立复现。没有官方GPU评测；提示词把search_similar_code用于自然语言查询，与当前按索引符号查邻近向量的实现不符。token数只是字符估计。

阅读定位：L1218–L1304, L1449–L1506, L1573–L1635, L1664–L1674。

## 46. Gemma 4 Developer Agent Top 1 Master Baseline

[Kaggle](https://www.kaggle.com/code/tharunkumar369/gemma-4-developer-agent-top-1-master-baseline) · [本地源码](public/tharunkumar369/gemma-4-developer-agent-top-1-master-baseline/readable.txt)

**类型：单 agent 提示词和静态打包。** 9个工具直接给coder；3.5分钟、32工具、40轮；自定义结构检查后打包。

可借鉴：短任务预算、精确小块编辑、语法检查与最终submit_patch。

边界与疑点：Top-1/SOTA和sweet-spot没有对应实测消融；temperature=0.15不能保证确定性，include_thoughts=false实际关闭thinking。检查器不验证工具参数，提示词示例仍需按官方签名检查。

阅读定位：L139–L178, L190–L240, L263–L307, L344–L351。

## 47. Gemma4 Day1 Baseline

[Kaggle](https://www.kaggle.com/code/tomaszszyrowski/gemma4-day1-baseline) · [本地源码](public/tomaszszyrowski/gemma4-day1-baseline/readable.txt)

**类型：最小打包与可选模型问答测试。** 生成两文件ZIP，可切换图工具；附本地Ollama问答和HF指定checkpoint加载/生成。

可借鉴：明确ZIP结构检查不替代官方validator，模型问答不替代修复闭环。

边界与疑点：RUN_HF_SMOKE实际为True，旁注却说disabled，会安装依赖/下载模型并要求单卡空闲32GiB。Ollama是不同runtime；未运行官方工具Agent或仓库测试。

阅读定位：L47–L83, L91–L130, L133–L180, L181–L206。

## 48. Gemma 4 Dev Agent: Zero to First Valid Submission

[Kaggle](https://www.kaggle.com/code/cocoyachi/gemma-4-dev-agent-zero-to-first-valid-submission) · [本地源码](public/cocoyachi/gemma-4-dev-agent-zero-to-first-valid-submission/readable.txt)

**类型：官方验证函数辅助的首次打包指南。** 安装wheelhouse、构建单agent、5分钟/40工具/100轮，调用模型声明和目录验证，计算预算后输出ZIP。

可借鉴：首次参与的目录结构与官方验证函数入口，比只检查ZIP存在更强。

边界与疑点：未实际执行GPU Agent。预算段把Phase2验证也写进12小时，与当前官方说明不一致；submit命令/Notebook入口以我们已验证File Upload流程为准。

阅读定位：L147–L176, L214–L271。

## 49. GTUA - Graph-Tool Utility Audit

[Kaggle](https://www.kaggle.com/code/norikazuhaji/gtua-graph-tool-utility-audit) · [本地源码](public/norikazuhaji/gtua-graph-tool-utility-audit/readable.txt)

**类型：图传播定位效用的基准审计。** TF-IDF定位加单跳传播，排除同文件及已经命中gold的seed后统计跨文件rescue，扫30组参数和仓库交互。

可借鉴：避免把原本已命中的种子误算图收益；冻结参数网格并保留降级结论。

边界与疑点：任务级p值不能证明跨仓库泛化；此实现不跑Agent或排行榜。是稳定性增强条目的基础版本，不应重复计为独立实验。

阅读定位：L180–L243, L294–L322。

## 50. notebookllms.io

[Kaggle](https://www.kaggle.com/code/harigov63/notebookllms-io) · [本地源码](public/harigov63/notebookllms-io/readable.txt)

**类型：全部写在 Markdown 中的模拟片段。** 正文定义mock_swegemma_inference，只打印阶段、sleep和返回固定补丁；三个单元都是Markdown。

可借鉴：识别展示代码、可执行单元及真实推理调用的区别。

边界与疑点：没有任何执行单元、模型加载、评测或提交ZIP；标题和打印的Framework online都不是运行证据。

阅读定位：L1–L37。

## 51. Gemma 4 SuperAgent

[Kaggle](https://www.kaggle.com/code/matterhorn3838/gemma-4-superagent) · [本地源码](public/matterhorn3838/gemma-4-superagent/readable.txt)

**类型：公开配置版本对照与再打包。** 默认原样输出top_bundle；custom才启用架构、issue提示、图工具和采样变更；保留每版字节hash及定制validator。

可借鉴：对改动做显式文本断言；区分默认原样配置和自定义实验。

边界与疑点：0.12/0.15等是引用作者自述，不是此notebook独立跑分。include_thoughts=false被误解释成仅隐藏思考，当前会关闭thinking；无eval_config时用harness默认预算。

阅读定位：L40–L100, L1851–L1895。

## 52.   Google - The Gemma 4 Developer 5

[Kaggle](https://www.kaggle.com/code/chaitanyajamble/google-the-gemma-4-developer-5) · [本地源码](public/chaitanyajamble/google-the-gemma-4-developer-5/readable.txt)

**类型：预算计算与带导航 skill 的打包脚本。** 构建coder+analyzer+源码定位skill；可复制已有adapter，末尾尝试swegemma CLI评测但失败不阻止打包。

可借鉴：不伪造adapter；清楚分离复制权重和实际训练，保存预算参数。

边界与疑点：MIN_TASK_MINUTES=5使自动预算可能超限：默认300题至少25小时；公开129题也不等于隐藏题数。CLI存在性不保证命令/模型服务就绪，check=False吞掉失败，成功打包不证明smoke通过。

阅读定位：L45–L66, L469–L505, L539–L566, L615–L644, L711–L766。

## 53. Gemma 4 Dev Agent: What Moves the Leaderboard

[Kaggle](https://www.kaggle.com/code/hitarthjain0/gemma-4-dev-agent-what-moves-the-leaderboard) · [本地源码](public/hitarthjain0/gemma-4-dev-agent-what-moves-the-leaderboard/readable.txt)

**类型：比赛机制导读与双 agent 打包。** 汇总工具/预算/失败模式，用依赖少的自定义validator构建analyzer+coder、8192输出且关闭thinking的配置。

可借鉴：便于团队理解提交契约和将实验日志与payload分开。

边界与疑点：“graph-first提升”和公开分数不是成对消融结论；LoRA尚不能工作等描述为旧状态，需当前wheel/官方回复核实。自定义结构检查不等于官方编译，更不等于得分。

阅读定位：L106–L123, L150–L184, L233–L286, L316–L332。

## 54. Gemma 4 AST Graph-Augmented SWE Multi-Agent System

[Kaggle](https://www.kaggle.com/code/avikdas567/gemma-4-ast-graph-augmented-swe-multi-agent-system) · [本地源码](public/avikdas567/gemma-4-ast-graph-augmented-swe-multi-agent-system/readable.txt)

**类型：EDA、双 agent 配置与可选真实评测混合。** 复制示例main/tool adapters，配置20分钟单题；服务成功时用Evaluator跑前两题，否则写空预测。

可借鉴：可参考真实Evaluator调用及结果字段，同时检查服务失败分支。

边界与疑点：adapter缺失时写512字节全零假safetensors，不是合法训练权重；必须移除该fallback。上下文曲线和转移矩阵是公式/常量模拟；T4×2叙述不代表官方L4×4资源，20分钟预算也不适合全量上限。

阅读定位：L783–L803, L868–L888, L945–L987, L1011–L1052, L1113–L1188。

## 55. GraphLoc end-to-end: failure analysis + resource

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-end-to-end-failure-analysis-resource) · [本地源码](public/zzgtylors/graphloc-end-to-end-failure-analysis-resource/readable.txt)

**类型：GraphLoc 已有运行的失败归因与方差分析。** 读取多次base/skill运行产物，定位patch到gold文件/函数，解析测试失败，做任务配对bootstrap并输出汇总资源。

可借鉴：保存trace、错误、真实patch和同题配对结果，区分定位错与修复错。

边界与疑点：当前notebook不重新推理；依赖挂载的完整运行输出。失败分类使用日志正则/耗时阈值，是启发式；任务bootstrap不自动处理仓库间泛化，不能把gold位置命中当resolved。

阅读定位：L1–L9, L92–L137, L167–L245, L318–L349。

## 56. GraphLoc end-to-end: repeat run 3

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-end-to-end-repeat-run-3) · [本地源码](public/zzgtylors/graphloc-end-to-end-repeat-run-3/readable.txt)

**类型：GraphLoc skill 的官方两阶段成对实验。** 实现离线BM25+重建图PageRank定位skill，与无skill配置共用采样预算；官方验证先筛gold通过任务，再让12个worker共享TP4模型执行两组Agent。

可借鉴：真实端到端对照、同题配对统计、重复运行及全trace记录，比单纯检索指标更接近目标。

边界与疑点：只按gold通过筛选，未证明empty必须失败；本地补依赖/修改Pygments入口有环境差异。12并发并非正式顺序评测耗时；三个公开条目源码相同，不代表三次结果已独立验证。

阅读定位：L294–L344, L968–L1021, L1088–L1158, L1194–L1209, L1399–L1452。

## 57. GraphLoc end-to-end: repeat run 2

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-end-to-end-repeat-run-2) · [本地源码](public/zzgtylors/graphloc-end-to-end-repeat-run-2/readable.txt)

**类型：GraphLoc skill 的官方两阶段成对实验。** 与条目 56（zzgtylors/graphloc-end-to-end-repeat-run-3）readable.txt 完全相同，已核对SHA256。实现离线BM25+重建图PageRank定位skill，与无skill配置共用采样预算；官方验证先筛gold通过任务，再让12个worker共享TP4模型执行两组Agent。

可借鉴：真实端到端对照、同题配对统计、重复运行及全trace记录，比单纯检索指标更接近目标。

边界与疑点：重复源码不构成独立的运行或得分证据。只按gold通过筛选，未证明empty必须失败；本地补依赖/修改Pygments入口有环境差异。12并发并非正式顺序评测耗时；三个公开条目源码相同，不代表三次结果已独立验证。

阅读定位：L294–L344, L968–L1021, L1088–L1158, L1194–L1209, L1399–L1452。

## 58. Gemma 4 | Budget-Aware Coding Agent

[Kaggle](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent) · [本地源码](public/flexonafft/gemma-4-budget-aware-coding-agent/readable.txt)

**类型：显式预算的双 agent 基线变体。** 保留原公开提示词采样，给coder增导航工具，设置4.5分钟/40工具/80轮；本地include路径、Apache许可及ZIP校验齐备。

可借鉴：保留上游来源许可，显式预算；有包时额外调用官方ADK schema。

边界与疑点：作者明确本candidate没有实测分数；可选schema会在无包时跳过，不能据打包完成声称官方验证通过。思考配置实际关闭，预算仍需L4吞吐测量。

阅读定位：L1–L44, L397–L447。

## 59. Gemma 4 Developer Agent CPU Audit Tables

[Kaggle](https://www.kaggle.com/code/nitish5236goel/gemma-4-developer-agent-cpu-audit-tables) · [本地源码](public/nitish5236goel/gemma-4-developer-agent-cpu-audit-tables/readable.txt)

**类型：历史发布审计与统计功效表重生成。** 嵌入原始记录与审计代码，重算gold符号/图结构/功效表，对比旧发布与9月25日重发，并核对来源commit的SHA。

可借鉴：明确哪些是本次重算、哪些是历史verifier/tool probe记录；检查测试不是空跑及所需case均通过。

边界与疑点：1210次+32次验证记录不会在此重新执行；历史空图结论不能套用当前数据。功效表依赖独立任务等模型假设，不是Agent得分或未来提分保证。

阅读定位：L1–L34, L869–L924, L1093–L1171, L3495–L3510, L3561–L3591, L3657–L3668。

## 60. G4CA_Basv1.0

[Kaggle](https://www.kaggle.com/code/ckskaggle/g4ca-basv1-0) · [本地源码](public/ckskaggle/g4ca-basv1-0/readable.txt)

**类型：受限双 agent 基线与自定义规则检查器。** coder保留导航并按需调用一次analyzer，提示词要求早改/短验证；构建ZIP前自定义检查，存在官方包时动态寻找验证函数。

可借鉴：把schema存在、结构检查和运行评测分开，记录提示词与工具权限。

边界与疑点：官方验证异常仅打印并继续，不能视作通过门槛；没有真实GPU评测或eval_config预算。提示词中的测试必被丢弃等假设需当前harness核实。

阅读定位：L37–L128, L249–L327。

## 61. TFD-Agent: Autonomous SWE-bench Resolution with Ge

[Kaggle](https://www.kaggle.com/code/rajrajak99/tfd-agent-autonomous-swe-bench-resolution-with-ge) · [本地源码](public/rajrajak99/tfd-agent-autonomous-swe-bench-resolution-with-ge/readable.txt)

**类型：论文式架构说明及模拟对象。** CodebaseEnvironment只在dict里读写字符串；展示四组硬编码分数和双agent流程图，没有Gemma调用和官方评测。

可借鉴：复现前检查图表的数值来源，分清宣称历史成绩与预测。

边界与疑点：0.13/排名为作者自述，M4明确projected，画图却填0.18；该值不是实验结果。没有可运行完整Agent或提交ZIP。

阅读定位：L1–L16, L59–L112。

## 62. Gemma 4 Developer Agent: End-to-End ADK Starter

[Kaggle](https://www.kaggle.com/code/koushikkumardinda/gemma-4-developer-agent-end-to-end-adk-starter) · [本地源码](public/koushikkumardinda/gemma-4-developer-agent-end-to-end-adk-starter/readable.txt)

**类型：不完整的 ADK 教学与打包草稿。** 检测空图后只累加restored_count，真实复制被注释；vLLM创建/启动也被注释；生成内联sub_agents/as_tool配置和ZIP。

可借鉴：可作为检查“有说明但无实际实现”的反例。

边界与疑点：“Restored”打印不代表修复发生；eval_config写在configs/而非根目录，harness未必读取。未用官方schema验证该sub_agent写法，不能称端到端运行成功。

阅读定位：L62–L85, L95–L115, L126–L172。

## 63. Can Gemma 4 Fix an Unseen Repo? | 31B QLoRA Study

[Kaggle](https://www.kaggle.com/code/denpugovkin/can-gemma-4-fix-an-unseen-repo-31b-qlora-study) · [本地源码](public/denpugovkin/can-gemma-4-fix-an-unseen-repo-31b-qlora-study/readable.txt)

**类型：真实训练代码与 oracle 定位的 QLoRA 研究。** Unsloth 31B bnb4bit、rank8、1024tokens单epoch SFT；按仓库留出Rich并mask提示词，保存PEFT权重，再成对生成base/adapter diff。

可借鉴：固定模型revision/依赖/切分；训练损失、草稿指标、artifact验证及正式Agent成功率分层记录。

边界与疑点：训练base不同于官方QAT；输入片段由gold diff定位，草稿验证是oracle条件，不测导航。git apply --numstat仅解析；未验证官方服务加载或resolved。代码有实际trainer.train，但我们未执行第三方训练。

阅读定位：L149–L170, L283–L299, L308–L388, L392–L415, L451–L518, L638–L708, L751–L769。

## 64. Pathfinder  | Gemma 4 Agent: EDA + Baseline+

[Kaggle](https://www.kaggle.com/code/mizeroluckygall/pathfinder-gemma-4-agent-eda-baseline) · [本地源码](public/mizeroluckygall/pathfinder-gemma-4-agent-eda-baseline/readable.txt)

**类型：Pathfinder 任务/检索分析与双 agent 打包。** 用issue符号比较name lookup、embedding邻近与词法搜索；v2生成coder+analyzer，默认不写eval_config，记录ZIP/hash。

可借鉴：根据具体符号研究检索效用；局部提示词、公开成绩和待测新版本分别列出。

边界与疑点：图研究只纳入可映射gold的任务且限60commit/300秒；v1的0.08为自述，v2 pending。include_thoughts=false会关闭thinking，不能按注释当仅隐藏。

阅读定位：L57–L71, L416–L472, L641–L697, L837–L849。

## 65. The Gemma 4 Developer Agent Competition

[Kaggle](https://www.kaggle.com/code/alfira17/the-gemma-4-developer-agent-competition) · [本地源码](public/alfira17/the-gemma-4-developer-agent-competition/readable.txt)

**类型：多版本草稿，最终 V12 双 agent 提交构建。** 前段构造单agent，后续单元多次覆盖submission.zip；最终从官方sample复制并移除adapters/eval_config，替换coder/analyzer提示词和8192采样。

可借鉴：阅读最终写出的ZIP；固定时间戳和5文件检查便于识别实际提交内容。

边界与疑点：前面四文件包不是最终payload；V12无自定义预算、无官方编译或GPU评测。YAML compose和字符串断言只能证明局部结构。

阅读定位：L207–L244, L303–L392。

## 66. GraphLoc end-to-end: Gemma 4 agent + skill

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-end-to-end-gemma-4-agent-skill) · [本地源码](public/zzgtylors/graphloc-end-to-end-gemma-4-agent-skill/readable.txt)

**类型：GraphLoc skill 的官方两阶段成对实验。** 与条目 56（zzgtylors/graphloc-end-to-end-repeat-run-3）readable.txt 完全相同，已核对SHA256。实现离线BM25+重建图PageRank定位skill，与无skill配置共用采样预算；官方验证先筛gold通过任务，再让12个worker共享TP4模型执行两组Agent。

可借鉴：真实端到端对照、同题配对统计、重复运行及全trace记录，比单纯检索指标更接近目标。

边界与疑点：重复源码不构成独立的运行或得分证据。只按gold通过筛选，未证明empty必须失败；本地补依赖/修改Pygments入口有环境差异。12并发并非正式顺序评测耗时；三个公开条目源码相同，不代表三次结果已独立验证。

阅读定位：L294–L344, L968–L1021, L1088–L1158, L1194–L1209, L1399–L1452。

## 67. 🔬 Gemma 4 Starter | Inside the Harness + E 63dd0e

[Kaggle](https://www.kaggle.com/code/ahmadalalony/gemma-4-starter-inside-the-harness-e-63dd0e) · [本地源码](public/ahmadalalony/gemma-4-starter-inside-the-harness-e-63dd0e/readable.txt)

**类型：通用starter的提示词迭代。** 与基线源码对比，仅降低max_output_tokens到4096并覆盖coder/analyzer提示词：强调工具调用、symbol查询、早编辑与失败搜索不重试。

可借鉴：用源码差异确认改动，修正自然语言图查询和/tmp文件工具假设。

边界与疑点：提示词固定pytest -p no:anyio可能改变异步测试行为，需按任务验证；环境工具存在性不能作为跨runtime事实。依旧只有静态打包，没有新得分证据。

阅读定位：L54–L60, L1306–L1371。

## 68. Gemma 4 Developer Agent: Direct Code Verifier

[Kaggle](https://www.kaggle.com/code/nihilisticneuralnet/gemma-4-developer-agent-direct-code-verifier) · [本地源码](public/nihilisticneuralnet/gemma-4-developer-agent-direct-code-verifier/readable.txt)

**类型：真实官方评测下的编辑工具候选对照。** rapid与guard用同一QAT服务，分两批各8题成对Evaluator；guard只留shell/read，不给edit_file，以总16题至少多解2题、零测试修改/基础设施错误选择提交。

可借鉴：逐题保留错误阶段与进度；提交仅含选中agent，外围评测记录不进入ZIP。

边界与疑点：候选同时改变工具、提示词、输出/思考预算，不能将差异归因单变量。第二批也参与选择，不是独立最终测试；没有empty/gold控制筛选。源码中启动真实评测不等于我们独立复现了结果。

阅读定位：L269–L310, L317–L357, L433–L455, L497–L578, L622–L640, L677–L718。

## 69. Gemma 4 DevAgent - KerasHub LoRA Training on TPU

[Kaggle](https://www.kaggle.com/code/nursrijan/gemma-4-devagent-kerashub-lora-training-on-tpu) · [本地源码](public/nursrijan/gemma-4-devagent-kerashub-lora-training-on-tpu/readable.txt)

**类型：KerasHub TPU 教学，但模型/adapter 与比赛不匹配。** 实际加载gemma2_instruct_2b_en并启用rank16 LoRA，随后把导出配置写成Gemma4 31B；加载失败则用64维随机矩阵构造safetensors。

可借鉴：可以了解TPU/Keras的一般训练接口；采用前必须核对底座架构和权重转换。

边界与疑点：Gemma2 2B不是规定的Gemma4 31B；改base_model_name不能转换架构，简单替换权重名也未处理PEFT名称/布局。随机fallback不是训练结果，不能作为比赛有效adapter方案。

阅读定位：L168–L203, L212–L245, L249–L301, L355–L369。

## 70. Black Cat SWE Agent 

[Kaggle](https://www.kaggle.com/code/harishyadav0506/black-cat-swe-agent) · [本地源码](public/harishyadav0506/black-cat-swe-agent/readable.txt)

**类型：Black Cat 单 agent 主包与备用包。** 静态字符串构建主包9工具和较简备用包，递归路径/扩展检查后输出两个ZIP。

可借鉴：主备包明确命名，符号查询和最小修复提示可作为对照。

边界与疑点：仅自定义结构验证；同时填写thinking_level/budget并设置include_thoughts=false，不能从MEDIUM标签认定思考开启。分数在源码明确pending。

阅读定位：L126–L128, L138–L198, L203–L214。

## 71. 🐈‍⬛ Black Cat SWE Agent | Second Strike

[Kaggle](https://www.kaggle.com/code/azouzibillel/black-cat-swe-agent-second-strike) · [本地源码](public/azouzibillel/black-cat-swe-agent-second-strike/readable.txt)

**类型：极简配置包与无关的模型变量。** 写单文件agent.yaml并打包；后续MODEL_ID另指旧Gemma2B但没有被载入或使用。

可借鉴：区分实际YAML模型声明和无效的notebook变量。

边界与疑点：没有编译/评测/训练；最后检查的是未生成的submission.json，与实际submission.zip不一致。不要把打印“就绪”当正式链路通过。

阅读定位：L1–L46。

## 72. 📘Gemma-4 | The Complete Guide. Expanded

[Kaggle](https://www.kaggle.com/code/leoprovorov/gemma-4-the-complete-guide-expanded) · [本地源码](public/leoprovorov/gemma-4-the-complete-guide-expanded/readable.txt)

**类型：官方 sample 原样打包控制。** 复制sample_submission所有文件并校验逐文件hash，make_archive后再逐文件读回比对。

可借鉴：作为打包阶段的严格内容一致性对照，记录原始sample版本。

边界与疑点：不加载GPU、不运行官方compiler/评测；sample含哪些adapter取决于挂载版本。文件内容相同不表示ZIP字节确定性或其模型已成功运行。

阅读定位：L1–L4, L24–L71。

## 73. notebook67dada2eb5

[Kaggle](https://www.kaggle.com/code/bang1850/notebook67dada2eb5) · [本地源码](public/bang1850/notebook67dada2eb5/readable.txt)

**类型：仅安装依赖的占位条目。** 只有一行pip安装TRL/PEFT/bitsandbytes等。

可借鉴：识别尚未实现的占位notebook。

边界与疑点：无固定版本、训练、模型、评测或提交构建；不能当RL/SFT教程已实现。

阅读定位：L1–L2。

## 74. GemmaForge Agent

[Kaggle](https://www.kaggle.com/code/salvapiol/gemmaforge-agent) · [本地源码](public/salvapiol/gemmaforge-agent/readable.txt)

**类型：大型 Agent 原型、轨迹模板和最终配置包。** 构建本地环境/trace记录类，把SWE-bench Lite转为模板步骤；ModelPolicy.generate仍未实现。最后拷贝官方示例adapters，写coder/analyzer/导航skill并打包。

可借鉴：区分工具环境实现、人工/模板轨迹、模型策略和最终ADK payload；可借鉴trace记录字段。

边界与疑点：生成模板步骤没有真实工具观测，不是成功Agent轨迹；LoRA结尾只建LoraConfig，没有训练。提交仍为复制示例权重，文件缺失只打印；中途项目ZIP也不是最终submission.zip。

阅读定位：L477–L554, L4799–L4836, L5161–L5173, L5196–L5218, L5861–L5888, L6030–L6104。

## 75. Google - The Gemma 4 Developer -11

[Kaggle](https://www.kaggle.com/code/chaitanyajamble/google-the-gemma-4-developer-11) · [本地源码](public/chaitanyajamble/google-the-gemma-4-developer-11/readable.txt)

**类型：明确资源上限的单 agent 打包。** 单agent9工具，8192输出，8分钟/30工具/40轮；自定义验证后写ZIP，并检查输出目录是否有其他文件。

可借鉴：短配置、include读回及打包错误显式化。

边界与疑点：8分钟乘约120题为16小时上界，未证明12小时总预算安全；“工作目录只能有ZIP”是作者约定，不是我们应额外删除日志的依据。无实际推理/得分。

阅读定位：L124–L154, L212–L233, L348–L359。

## 76. GEMMA 4 developer agent EDA

[Kaggle](https://www.kaggle.com/code/msama01/gemma-4-developer-agent-eda) · [本地源码](public/msama01/gemma-4-developer-agent-eda/readable.txt)

**类型：通用 EDA starter 的配置适配版本。** 解析gold diff/词法定位，读取sample的根配置名与model，生成单agent或coder+analyzer并打包。

可借鉴：跟踪变量最终值：早期swe_coder.yaml会被后面的sample发现逻辑覆盖，不应仅凭首个赋值误判最终包。

边界与疑点：仍为EDA+自定义validator，没有官方GPU评测。示例数据fallback/字符token估计不应误写成真实测量，thinking配置实际关闭。

阅读定位：L7–L40, L387–L421, L1366–L1394, L1408–L1489。

## 77. Gemma 4 Developer Agent

[Kaggle](https://www.kaggle.com/code/harishyadav0506/gemma-4-developer-agent) · [本地源码](public/harishyadav0506/gemma-4-developer-agent/readable.txt)

**类型：早期比赛机制说明与最小单 agent 包。** 构建6核心工具的单agent，temperature0/top_k1且thinking开启，提示词按约10工具/短任务预算规划；结构检查后输出ZIP。

可借鉴：可读的单agent基线，明确最终产物应是可执行配置。

边界与疑点：原文256组图及默认极短预算属于旧材料；当前图已核验127组。不能把pytest退出0单独当完备成功判据，仍需测试存在/目标覆盖等官方verifier条件；无实际得分。

阅读定位：L26–L44, L229–L280, L350–L428, L484–L508。

## 78. 🐈‍⬛ Gemma 4 Agent: EDA & ADK Starter

[Kaggle](https://www.kaggle.com/code/nilanjansaha123/gemma-4-agent-eda-adk-starter) · [本地源码](public/nilanjansaha123/gemma-4-agent-eda-adk-starter/readable.txt)

**类型：简化EDA和双 agent 配置打包。** 解析diff，尝试按instance_id读取图，创建coder+analyzer并写ZIP。

可借鉴：配置较短，方便理解agent_tool结构和独立提示词。

边界与疑点：图路径只找任务名，不覆盖当前commit命名，可能静默不画图。所谓ADK schema验证实际仅检查扩展/路径；没有官方编译，include_thoughts=false与HIGH标签不能证明开启thinking。

阅读定位：L89–L111, L117–L171, L177–L190。

## 79. GraphLoc: auditing Gemma 4 agent code graphs

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-auditing-gemma-4-agent-code-graphs) · [本地源码](public/zzgtylors/graphloc-auditing-gemma-4-agent-code-graphs/readable.txt)

**类型：GraphLoc 图/embedding/定位 CPU 研究。** 从snapshot AST与参考diff生成函数标签，比较G0/G+、BM25、图传播、embedding及外部SWE-bench迁移，导出stdlib定位skill。

可借鉴：同等候选查看预算、明确gold标签来源、将图结构覆盖与定位效果分开。

边界与疑点：不跑修复Agent；标签和排名不是resolved。末尾并发启动多项批处理，明显不适合登录节点Run All；外部迁移准备需要网络/仓库预备。

阅读定位：L1–L12, L1130–L1187, L1760–L1778, L1781–L1809, L1998–L2019, L2104–L2133。

## 80. Google - The Gemma 4 Developer/

[Kaggle](https://www.kaggle.com/code/chaitanyajamble/google-the-gemma-4-developer) · [本地源码](public/chaitanyajamble/google-the-gemma-4-developer/readable.txt)

**类型：可选SFT与三层fallback的提交构建。** 默认RUN_LORA_TRAINING=false；若打开则训练issue→diff LoRA。打包优先sample，失败转内建agent，再失败输出emergency单文件。

可借鉴：默认无adapter的行为清晰；检查最终产物标签和hash以知道实际采用了哪个分支。

边界与疑点：训练失败会被吞掉继续无adapter，emergency返回未承接完整验证；“总能有ZIP”不证明预定训练/配置成功。随机题切分也不代表新仓库泛化；需独立官方运行。

阅读定位：L383–L435, L518–L550, L1055–L1085, L1187–L1202。

## 81. notebook13918923cc

[Kaggle](https://www.kaggle.com/code/manuelgagofernandez/notebook13918923cc) · [本地源码](public/manuelgagofernandez/notebook13918923cc/readable.txt)

**类型：Kaggle 默认脚本模板。** 仅导入numpy/pandas/kagglehub并打印挂载文件路径。

可借鉴：保留空白开发起点的覆盖记录。

边界与疑点：没有本比赛实现、模型、评测或产物；与同作者另一模板内容近同。

阅读定位：L1–L24。

## 82. 🧬 Anatomy of a Fix

[Kaggle](https://www.kaggle.com/code/proamari/anatomy-of-a-fix) · [本地源码](public/proamari/anatomy-of-a-fix/readable.txt)

**类型：问题—参考修复—测试的 EDA 与提示词包。** 围绕129题讲解issue、gold patch和test_patch，统计线索/改动并生成coder+analyzer配置与hash。

可借鉴：团队介绍素材和按证据组织设计假设；正确检查实际导入工作树。

边界与疑点：没有运行Agent或证明数据驱动提示词提分。固定禁用anyio和“永不重复命令”可能妨碍必要复测，不能照搬；temperature等“所有有效提交都相同”也不是可验证普遍事实。

阅读定位：L14–L25, L169–L183, L302–L323, L329–L342, L376–L421。

## 83. Gemma 4 Agent Working Baseline

[Kaggle](https://www.kaggle.com/code/leoprovorov/gemma-4-agent-working-baseline) · [本地源码](public/leoprovorov/gemma-4-agent-working-baseline/readable.txt)

**类型：通用starter的另一提示词变体。** 与inside-the-harness条目逐行比对：输出上限8192，增加导入路径确认，改变复现/验证指令及重复命令约束。

可借鉴：只依据实际源码差异列出实验变量；导入路径验证有助排查假修复。

边界与疑点：没有新GPU结果；“永不重复同一命令”与修复后重跑验证存在张力，pytest -p no:anyio需按任务判断。它不是完全相同源码的独立运行复现。

阅读定位：L54–L60, L1308–L1371。

## 84. Google - The Gemma 4 Developer Agent Competition

[Kaggle](https://www.kaggle.com/code/manuelgagofernandez/google-the-gemma-4-developer-agent-competition) · [本地源码](public/manuelgagofernandez/google-the-gemma-4-developer-agent-competition/readable.txt)

**类型：Kaggle 默认 Python 模板。** 仅导入包并列出输入文件，无比赛业务逻辑。

可借鉴：完整覆盖公开条目，标识未实现内容。

边界与疑点：没有Agent/训练/模型服务/ZIP，不能从比赛标题推断完成度。

阅读定位：L1–L23。

## 85. GraphLoc: can Gemma 4 pick the code to change?

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-can-gemma-4-pick-the-code-to-change) · [本地源码](public/zzgtylors/graphloc-can-gemma-4-pick-the-code-to-change/readable.txt)

**类型：官方底座的候选代码重排实验。** 从BM25和一跳图构造top10/hop1/同长度same候选池，Gemma TP4选top3，比较pool覆盖、选择准确率、thinking与配对bootstrap。

可借鉴：控制候选数量与顺序，区分候选池遗漏和模型选错；保存prompt/output token数。

边界与疑点：只包含有可映射gold函数的题；代码定位不是补丁修复。chat参数TypeError会fallback不显式传thinking开关，需记录实际render状态；未由我们独立跑GPU。

阅读定位：L1–L12, L80–L127, L148–L191, L213–L240。

## 86. Gemma 4 Paper Track Async Codegraph Resource

[Kaggle](https://www.kaggle.com/code/sayandeepghosh/gemma-4-paper-track-async-codegraph-resource) · [本地源码](public/sayandeepghosh/gemma-4-paper-track-async-codegraph-resource/readable.txt)

**类型：AsyncPolyGraph 原型与有标签偏差的定位审计。** 定义async/调用/装饰/依赖边builder，但评测主函数只读已有图和按路径词法打分；候选文件catalog取自所有任务gold patch。

可借鉴：可研究异步函数、依赖注入边的表示，但要另写不依赖答案的候选生成。

边界与疑点：builder未用于所报告对比；gold文件进入候选全集造成oracle信息，不能作为无泄漏定位性能。按instance_id查图也未覆盖当前commit文件；报告提升不证明async图有效。

阅读定位：L41–L58, L102–L126, L151–L189, L197–L258, L286–L311。

## 87. Gemma4 RQ4 Query Encoding

[Kaggle](https://www.kaggle.com/code/woldywei/gemma4-rq4-query-encoding) · [本地源码](public/woldywei/gemma4-rq4-query-encoding/readable.txt)

**类型：查询文本编码与已发布embedding通道的定位对照。** Jina-code失败则BGE-small，编码issue/节点文本，比较TF-IDF、索引embedding和RRF，报告文件/函数Recall与MRR。

可借鉴：识别当前search_similar_code无查询文本编码器；分别量化query通道和候选覆盖。

边界与疑点：其emb_rank按token交集平均多seed，并非严格复刻官方exact/suffix查询。gold使用新diff行号映射原始AST可能错位；模型fallback改变实验，需分开标记。未跑修复Agent，第三方encoder不能直接假定正式提交允许。

阅读定位：L1–L13, L183–L211, L267–L290, L399–L454。

## 88. Gemma | Super Basic

[Kaggle](https://www.kaggle.com/code/twangygarlic449/gemma-super-basic) · [本地源码](public/twangygarlic449/gemma-super-basic/readable.txt)

**类型：最小五工具单 agent scaffold。** 写root、sampling和prompt三文件，5个工具、8192输出、thinking关闭，检查必需文件后打包。

可借鉴：便于隔离最基本的提示词变化与包结构。

边界与疑点：无官方编译或推理；没有eval_config，取当前harness默认预算。仅检查文件存在/根配置，不证明工具调用能成功。

阅读定位：L17–L31, L66–L85, L102–L157。

## 89. Getting Started - Gemma 4 Developer Agent

[Kaggle](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent) · [本地源码](public/ryanholbrook/getting-started-gemma-4-developer-agent/readable.txt)

**类型：官方端到端入门。** 离线安装定制 wheels，复制 sample 并启用 thinking，启动 vLLM v2/32768 context；官方 Evaluator 顺序执行前两题的生成和验证，最后打包 agent。

可借鉴：作为模型注册、adapter discovery、两阶段评测与 ZIP 根目录的权威实现参照。

边界与疑点：两题示例不是全量成绩；复制 sample adapter 不代表训练完成。强制重装和删除 cutlass .pth 只适合隔离运行环境；必须另加 empty/gold 控制和实际资源核验。

阅读定位：L26–L98, L167–L203, L239–L320。

## 90. Gemma4 Localization Repaired

[Kaggle](https://www.kaggle.com/code/woldywei/gemma4-localization-repaired) · [本地源码](public/woldywei/gemma4-localization-repaired/readable.txt)

**类型：重建图后的定位实验。** 从 snapshot AST 重建含 async 定义的 calls 图，以 TF-IDF、已有 embedding 的符号种子均值、PPR 和 RRF 比较文件/函数召回。

可借鉴：将图覆盖与排序能力分开，检查 async 定义、名称解析和图/embedding 对齐。

边界与疑点：重建图未同步重建 embeddings；新文件 hunk 行号与旧 snapshot AST 对齐可能错标。种子均值不是官方 search_similar_code 的完全复现，定位分数不是修复成功率；tar 批量展开不适合登录节点。

阅读定位：L149–L170, L195–L264, L322–L362, L477–L575。

## 91. Gemma4 Repaired Graphs Regen

[Kaggle](https://www.kaggle.com/code/woldywei/gemma4-repaired-graphs-regen) · [本地源码](public/woldywei/gemma4-repaired-graphs-regen/readable.txt)

**类型：async 图生成工具。** 逐题解开 snapshot，AST 收集同步/异步函数及类，以名称与模块启发式连 calls 边，输出 instance_id.json 和 regen_summary。

可借鉴：可作为基于源码重建 async 覆盖的原型与图统计参考。

边界与疑点：没有训练、推理、修补或 embedding 更新。连边依赖名称启发式，不能视为精确动态调用图；输出任务名与当前官方 commit 名需要映射，解包必须隔离且检查路径。

阅读定位：L69–L90, L138–L184, L193–L222。

## 92. Gemma4 Graph Localization Bench

[Kaggle](https://www.kaggle.com/code/woldywei/gemma4-graph-localization-bench) · [本地源码](public/woldywei/gemma4-graph-localization-bench/readable.txt)

**类型：原始图定位基准。** TF-IDF 文件/函数检索、符号种子 embedding、PPR、RRF 在公开任务比较 recall/MRR，并报告 gold 在图节点中不可达的比例。

可借鉴：先测覆盖上限，再做相同候选空间的排序比较，并按仓库和 async 类型分层。

边界与疑点：函数表只汇总可达 gold，不能忽略未覆盖任务；使用新 hunk 行号对应旧代码存在标签偏差。embedding 查询为多符号种子近似，不是自然语言编码或官方精确复现；无 repair 推理。

阅读定位：L92–L128, L229–L275, L389–L445, L480–L504。

## 93. Gemma4 Graph Coverage Audit

[Kaggle](https://www.kaggle.com/code/woldywei/gemma4-graph-coverage-audit) · [本地源码](public/woldywei/gemma4-graph-coverage-audit/readable.txt)

**类型：图覆盖静态审计。** 按任务名、commit 名及共享 commit 查可读图，扫描参考 patch 触及文件内的定义，比较 graph 节点覆盖和 async 类别。

可借鉴：提供数据映射、空文件排查和 AST 覆盖诊断思路。

边界与疑点：计数对象是 gold 触及文件内所有定义，而非准确被改函数；末段同名匹配可能误判覆盖。旧 hardlink 描述不代表当前包仍损坏，snapshot 前缀处理需在当前文件布局复核。

阅读定位：L111–L138, L143–L182, L185–L227。

## 94. Gemma and the Shape of Doubt

[Kaggle](https://www.kaggle.com/code/prvsiyan/gemma-and-the-shape-of-doubt) · [本地源码](public/prvsiyan/gemma-and-the-shape-of-doubt/readable.txt)

**类型：有完整证据组织的 agent 与开发实验。** 保留作者已提交 V10 ZIP，官方 compiler 编译，另建有界源码搜索候选；包括固定补丁 CPU 重放和真实 vLLM/官方 verifier 的16题开发筛选。当前 RUN_COMMAND_SCREEN=True 会启用 GPU 路径。

可借鉴：借鉴 hash 固定提交、候选单独保存、端点健康检查、严格 JUnit、环境修复与 agent 收益分开归因。

边界与疑点：0.08、2/4、3/12 等为作者保留的历史记录，非我们复现或当前榜首。多组数据反复用于开发；V14 多出的通过与环境修复重合，不能归因于策略。默认不是纯 CPU 重打包；额外 wheels 和测试清单也必须附加。

阅读定位：L9–L33, L845–L904, L955–L985, L2700–L2745, L2849–L2875, L3773–L3791。

## 95. mainmainmain

[Kaggle](https://www.kaggle.com/code/tuyennguyen21pt/mainmainmain) · [本地源码](public/tuyennguyen21pt/mainmainmain/readable.txt)

**类型：与修复赛题无直接关系的情感分析训练。** 配置越南语 APC 数据、mBERT/mT5，调用外部 KLTN-Token-Merging 项目的 ATE/APC 多方法训练，并分析 span 错误。

可借鉴：仅可参考一般实验目录、错误分析与多种子组织。

边界与疑点：被 competition Code 页面收录不代表其任务或模型获准用于本赛；不是 Gemma agent、ADK bundle 或本赛评测。

阅读定位：L56–L84, L449–L480, L525–L550。

## 96. The Gemma 4 Developer Agent- Google

[Kaggle](https://www.kaggle.com/code/chaitanyajamble/the-gemma-4-developer-agent-google) · [本地源码](public/chaitanyajamble/the-gemma-4-developer-agent-google/readable.txt)

**类型：可选 SFT 与三层回退打包。** 默认 RUN_LORA_TRAINING=False；sample 改写失败后退到内置 agent，再退到 emergency 单文件。训练开关开启后可生成 adapter 文件。

可借鉴：保留最终实际使用的 package_label、ZIP 回读和未引用 adapter 的警告。

边界与疑点：自身 YAML/文件检查不是官方 compiler；训练后 adapter 仅加入文件，代码明确未接入 agent 配置。成功写出 ZIP 可能是 emergency 行为，不能视为预期 agent 已运行。

阅读定位：L377–L387, L504–L531, L793–L814, L842–L866。

## 97. notebook057a67b5a8

[Kaggle](https://www.kaggle.com/code/tuynnguynee/notebook057a67b5a8) · [本地源码](public/tuynnguynee/notebook057a67b5a8/readable.txt)

**类型：情感分析训练/错误分析副本变体。** 使用与 mainmainmain 相同的 APC 数据和 mBERT/mT5 配置，含 ATE 训练产物及 span 错误统计。

可借鉴：作为页面存在无关关联 notebook 的证据，不纳入比赛策略比较。

边界与疑点：无关模型的出现不能解释为本赛允许替代正式 Gemma 底座；与 index94 不是完整源码相同的重复项。

阅读定位：L56–L84, L449–L480。

## 98. Gemma 4 DevAgent - Graph Reasoning and SWE Engine

[Kaggle](https://www.kaggle.com/code/nursrijan/gemma-4-devagent-graph-reasoning-and-swe-engine) · [本地源码](public/nursrijan/gemma-4-devagent-graph-reasoning-and-swe-engine/readable.txt)

**类型：EDA、模拟图与 graph navigator 打包。** 分析 patch、演示弹性编辑；读取任务名图失败时合成 FastAPI 图，输出 navigator+coder YAML 并做文件级自检。

可借鉴：可以参考修改前后检查、符号查询、局部测试和职责拆分的提示词。

边界与疑点：当前官方图按 commit 命名，此任务名查询可能直接落入 mock；标注 Formal adk-submission 的函数实际只检查 root/模型数量/扩展/大小。没有真实 evaluator，include_thoughts=False 会关闭 thinking。

阅读定位：L250–L286, L438–L510, L544–L584。

## 99. 119 of 129 sound: the Gemma 4 grader, rebuilt

[Kaggle](https://www.kaggle.com/code/busyaprime/119-of-129-sound-the-gemma-4-grader-rebuilt) · [本地源码](public/busyaprime/119-of-129-sound-the-gemma-4-grader-rebuilt/readable.txt)

**类型：数据/验证器审计与词法定位研究。** 现场清点数据与 task 字段、按仓库时间分 dev/heldout，批量扫描 snapshots 测 BM25 等定位；验证 sound 数从历史审计表或附加 CSV 读取，现场重算统计。

可借鉴：借鉴 live 与 cited 的来源区分、基线失败/参考通过、required-node/JUnit 条件、固定切分与定位指标。

边界与疑点：标题119不是本 notebook 重跑129题验证所得；raw required-node surrogate 是117，119含对 skip 规则的作者解释。历史“无 harness”仅检查比赛数据目录，不能否定后来单独发布的官方 wheelhouse；不能据此断言当前评测绝对无法离线运行。

阅读定位：L1–L21, L677–L710, L767–L807, L835–L877, L1018–L1032, L1765–L1799。

## 100. Google - The Gemma 4 Developer Agent 2

[Kaggle](https://www.kaggle.com/code/chaitanyajamble/google-the-gemma-4-developer-agent-2) · [本地源码](public/chaitanyajamble/google-the-gemma-4-developer-agent-2/readable.txt)

**类型：EDA/仓库先验及可选 LoRA 打包器。** 默认汇总公开 patch 的仓库级常见修改文件并注入提示词；可选 issue→diff SFT，mask prompt、超长样本丢弃、梯度累积；adapter 需显式参数，训练失败则退到无 adapter。

可借鉴：参考模块化 CLI、真正标签掩码和对最终 bundle 回读验证。

边界与疑点：公开 gold 文件频率已进入提示词，不能拿同一公开集当无泄漏验证；隐藏仓库不同，先验泛化未知。SFT 无仓库上下文/独立切分，最后不足 grad_accum 的梯度没有显式 flush，量化底座和官方服务兼容性未证实。

阅读定位：L358–L388, L797–L857, L865–L883, L935–L967。

## 101. Gemma4

[Kaggle](https://www.kaggle.com/code/rishavae/gemma4) · [本地源码](public/rishavae/gemma4/readable.txt)

**类型：三 agent YAML 打包草稿。** 仅写 coder、analyzer、repro_verifier 的 YAML、提示词和 sampling，压缩成 submission.zip。

可借鉴：展示定位、修改、验证职责拆分的概念。

边界与疑点：标题宣称运行 GENSL/EC-PPR 和官方验证，但全部78行中没有数据读取、求解器或 compiler/evaluator 调用；嵌套 AgentTool schema、顶层 thinking_budget 字段未验证，自然语言 search_similar_code 查询也需改正。

阅读定位：L1–L78。

## 102. Dual-Space Spectral Graph SWE Agent for Gemma 4

[Kaggle](https://www.kaggle.com/code/avikdas567/dual-space-spectral-graph-swe-agent-for-gemma-4) · [本地源码](public/avikdas567/dual-space-spectral-graph-swe-agent-for-gemma-4/readable.txt)

**类型：双空间检索演示与占位 adapter。** 比较余弦与图距离加权检索，query 为目标 embedding 加噪，hybrid 直接使用目标符号作为 seed；打包时随机 LoRA A、全零 B。

可借鉴：仅可参考图/向量组合的教学结构和文件审计思路。

边界与疑点：检索实验提前知道目标，并非 issue 定位的可信效果；不同模式也不保证同一噪声 query。adapter 无训练、维度/层名手写，safetensors 格式成立不等于模型兼容或有收益；没有官方修复评测。

阅读定位：L1153–L1223, L1736–L1767, L1772–L1800。

## 103. Gemma 4 | Walkthrough & First Submission

[Kaggle](https://www.kaggle.com/code/zhukovoleksiy/gemma-4-walkthrough-first-submission) · [本地源码](public/zhukovoleksiy/gemma-4-walkthrough-first-submission/readable.txt)

**类型：版本对照导读与基线构建。** 与条目 8（liujiahua2358/gemma-4-v0-walkthrough-first-submission）readable.txt 完全相同，已核对SHA256。解析 sample、示例 LoRA、graph/embedding，并构建 analyzer+coder 的迭代版本；末尾打包提交。

可借鉴：读取每次版本变更及其来源；检查 graph 输出大小、正确读取测试尾部和固定 wheel hash。

边界与疑点：重复源码不构成独立的运行或得分证据。同一文档含旧版与后续修订说法，thinking/seed 转发须以当前 wheel 源码为准；引用的榜单分数不是其独立消融证明。

阅读定位：L1–L15, L275–L318, L380–L410, L630–L653。

## 104. Gemma4_DevAgent_GraphTools_Submission_v2

[Kaggle](https://www.kaggle.com/code/yuki16/gemma4-devagent-graphtools-submission-v2) · [本地源码](public/yuki16/gemma4-devagent-graphtools-submission-v2/readable.txt)

**类型：本地图工具实现、短导航循环与自检打包。** 实现图工具及 budget/错误 schema；缺 embedding 可用 hashing fallback。可用时加载 vLLM/Transformers 跑最多2轮导航，失败跳过；构建并自检 ZIP，缺 adapter 则去掉引用。

可借鉴：借鉴清晰记录 skipped/failed 状态、ZIP 临时文件校验后发布、非训练权重不引用。

边界与疑点：两轮只定位，不生成补丁或跑官方 verifier；工具返回作为 user 文本且用启发式解析，不等于官方 ADK 执行。hashing fallback 不等同官方 embedding，Notebook finished normally 也允许模型跳过。

阅读定位：L300–L323, L1014–L1047, L1072–L1118, L1723–L1737, L1740–L1790。

## 105. Gemma 4 Agent 🌟EDA + Agent + Validator

[Kaggle](https://www.kaggle.com/code/mizeroluckygall/gemma-4-agent-eda-agent-validator) · [本地源码](public/mizeroluckygall/gemma-4-agent-eda-agent-validator/readable.txt)

**类型：Pathfinder 的 EDA/定位与 analyzer+coder。** 默认 analyzer+coder，图实验最多60 commits/300秒；coder 提示4分钟软停，analyzer提示最多8次调用，打包后自写 validator。

可借鉴：明确从 exact symbol/grep 开始，复用同一复现命令验证，保留文件和源码行定位。

边界与疑点：soft_limit_seconds 仅进入提示词，默认 ship_eval_config=False，不是4分钟硬限制；8次子agent调用同样不是独立硬预算。include_thoughts=False 的“只隐藏推理”注释不符合当前 harness 行为；没有端到端 GPU 修复证据。

阅读定位：L56–L71, L575–L627, L717–L770。

## 106. GEMMA: EDA, Baseline for a start | LB TOP 1

[Kaggle](https://www.kaggle.com/code/romanrozen/gemma-eda-baseline-for-a-start-lb-top-1) · [本地源码](public/romanrozen/gemma-eda-baseline-for-a-start-lb-top-1/readable.txt)

**类型：EDA、词法定位与可选双 agent 打包。** 与条目 45（ahmadagbaria/gemma-4-baseline-reproduction）readable.txt 完全相同，已核对SHA256。解析数据和参考diff，做TF-IDF定位；构造单agent或analyzer+coder，定制validator、确定性ZIP及experiment_card。

可借鉴：源码/配置/hash一并留档；分离定位指标与最终提交内容。

边界与疑点：重复源码不构成独立的运行或得分证据。源码与romanrozen条目逐字节相同，并不是第二次独立复现。没有官方GPU评测；提示词把search_similar_code用于自然语言查询，与当前按索引符号查邻近向量的实现不符。token数只是字符估计。

阅读定位：L1218–L1304, L1449–L1506, L1573–L1635, L1664–L1674。

## 107. [0.10]: Gemma 4 Developer Agent Submission

[Kaggle](https://www.kaggle.com/code/nihilisticneuralnet/0-10-gemma-4-developer-agent-submission) · [本地源码](public/nihilisticneuralnet/0-10-gemma-4-developer-agent-submission/readable.txt)

**类型：lean/reviewer/full prompt 消融骨架。** 默认 lean 去掉两个子agent，4.5分钟/40工具/100turn，创建三种 ZIP；外部候选搜索、消融和 local eval 均默认关闭。

可借鉴：参考按相同任务比较精简 agent 与可选 reviewer，以及重复调用轨迹统计。

边界与疑点：标题0.10是作者宣称；默认未运行 evaluator。固定 pytest -p no:anyio -o timeout=0 会改变测试环境，应逐仓库核验，不能照搬；include_thoughts=False 实际禁用 thinking。

阅读定位：L294–L300, L353–L390, L454–L500, L595–L614, L672–L687。

## 108. Google - The Gemma 4

[Kaggle](https://www.kaggle.com/code/evgendvorkin/google-the-gemma-4) · [本地源码](public/evgendvorkin/google-the-gemma-4/readable.txt)

**类型：带俄语说明的 EDA 与 agent starter。** 默认 analyzer+coder、8192 output，分析 tasks/graphs/embedding/词法定位；按 sample 的 schema 构建，ZIP 回读并生成 experiment_card。

可借鉴：借鉴配置/hash/归档回读的实验记录和可视化解释。

边界与疑点：time_budget_s=240 是分析循环软限，不是 agent 硬预算；validator 自写，无官方推理成绩。include_thoughts=False 禁用 thinking；文件/字符估算不能证明实际上下文占用。

阅读定位：L285–L308, L1721–L1759, L1884–L1900, L1976–L1998。

## 109. 🐈‍⬛ Black Cat SWE Agent | Pack Instinct

[Kaggle](https://www.kaggle.com/code/lucifer19/black-cat-swe-agent-pack-instinct) · [本地源码](public/lucifer19/black-cat-swe-agent-pack-instinct/readable.txt)

**类型：保留已评分 anchor 的词法检索候选。** 默认选择 v13 challenger，静态解码三个历史 ZIP并验 hash；新 skill 对源码按标识符/错误文本/行为/full issue 多视图 BM25+RRF 排名，最多6000文件/12000chunks/32MiB，失败退到原 anchor。

可借鉴：保留已评分 ZIP、不覆盖恢复包、候选独立命名；bounded 源码搜索可作为后续消融对象。

边界与疑点：0.12属于历史 v10，非默认v13。内嵌 pilot 记录为服务器600秒启动超时、0 completed pairs；本 notebook 不重跑官方 compiler/GPU。analyzer 对 search_similar_code 的自然语言说明仍不合实际符号查询行为。

阅读定位：L1–L25, L128–L160, L176–L219, L225–L239。

## 110. Gemma4 Agent Advanced EDA

[Kaggle](https://www.kaggle.com/code/foysalemonshanto/gemma4-agent-advanced-eda) · [本地源码](public/foysalemonshanto/gemma4-agent-advanced-eda/readable.txt)

**类型：较完整的数据/上下文成本 EDA。** 检查 task/patch/test_patch、图和 embedding 两套命名、向量结构；批量扫描 snapshot，以 gold 触及文件及行区间估计全文读取与定点读取成本。

可借鉴：参考当前 commit 名映射、allow_pickle=False、按仓库切分和工具返回长度影响分析。

边界与疑点：使用 gold 定位的读取次数是已知答案条件下的成本参照，不是 agent 性能；没有训练或修复评测。ProcessPoolExecutor(max_workers=os.cpu_count()) 会大量并行扫描，不可直接在登录节点 Run All。

阅读定位：L414–L440, L556–L581, L659–L704, L711–L741。

## 111. Gemma 4 SWE Agent: Complete EDA & ADK Starter Kit

[Kaggle](https://www.kaggle.com/code/nursrijan/gemma-4-swe-agent-complete-eda-adk-starter-kit) · [本地源码](public/nursrijan/gemma-4-swe-agent-complete-eda-adk-starter-kit/readable.txt)

**类型：EDA 与双 agent 打包教学。** 读 tasks，缺数据时合成两条演示；模拟 edit_file，写 analyzer+coder YAML，自检文件大小/模型/扩展并输出 ZIP。

可借鉴：仅参考基本目录和精简局部修改/验证提示词。

边界与疑点：最后名为 placeholder 的 submission.parquet 实际直接填 row.patch，即 gold 答案，不是模型预测。无真实 evaluator，mock 数据和图不能当真实统计，include_thoughts=False 禁用 thinking。

阅读定位：L96–L135, L382–L413, L446–L478, L502–L539, L556–L598。

## 112. Gemma develop first try

[Kaggle](https://www.kaggle.com/code/haideptry/gemma-develop-first-try) · [本地源码](public/haideptry/gemma-develop-first-try/readable.txt)

**类型：官方 sample adapter 与5分钟双 agent 打包。** 将 sample adapters 或内嵌 base64 权重打包到 coder/analyzer；明确5分钟/15工具/25turn/60秒命令限，固定ZIP时间戳并回读。已静态解码内嵌权重检查 header和数据。

可借鉴：参考可复现 ZIP、当前 language_model 层路径与真正非方形 q/o tensor 形状。

边界与疑点：两个内嵌权重完全相同，LoRA B 全零；未训练，不能宣称获得训练收益或100%运行兼容。5×120=10小时只约束任务生成，不覆盖 setup，不能数学保证全赛时限；禁止 skills 是作者选择非官方规则。

阅读定位：L5–L11, L35–L65, L98–L124, L163–L189。
