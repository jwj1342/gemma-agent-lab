# 从 Agent 配置到正式提交

当前已通过**本地打包、私有 Kaggle CPU notebook 输出、官方编译、H100 与 Kaggle L4×4
单题真实推理、独立仓库测试，以及正式比赛评分**。同一份 ZIP 的正式提交 **56844353**
状态为 **COMPLETE**，公开分数 **0.12（12%）**，无评分错误；private 分数尚未提供。
查询时间为 2026-10-05 18:14 UTC。完整链路已跑通，但这不是训练或全量本地验证的完成声明。

## 1. 修改配置并构建 ZIP

基线源目录为 `configs/baseline/`：根 agent、analyzer、提示词、采样与预算配置。
使用指定模型，无 LoRA；示例 adapter 不作为已训练模型使用。

```bash
python scripts/build_submission.py \
  --agent-dir configs/baseline \
  --output runs/pipeline-baseline/submission.zip
```

构建器检查根 `agent.yaml`、文件扩展、符号链接、数量、展开大小和 ZIP CRC；固定文件
顺序、时间戳和权限，使相同配置构建出相同字节。它不替代官方 compiler。
旁边的 `submission.manifest.json` 包含逐文件 hash 和 archive hash。

当前 v1 ZIP SHA256：`02bb939095ded0c970088fc9d19634e203e2d3119a36a5feff20bad55c9343fc`。
修改配置后必须重新生成，不能继续沿用旧 hash。

## 2. 通过官方编译与真实任务检查

`scripts/validate_agent.py` 调用官方 `compile_submission`、比赛工具 registry 和 limits。
它不启动模型；需要先准备比赛环境。`scripts/evaluate_agent.py` 分离推理与验证：
推理只读取脱敏任务，验证另行读取 gold/test patch。

2026-10-05 官方 compiler 在本地隔离容器中已通过，模型别名、工具 registry 和
比赛 limits 均由官方包检查；最新证据为 `runs/hpc-smoke-20261005-r4/compile/validation.json`。
同一 run 的 `rich_3454` 未修复对照为 1 failed / 82 passed，gold 为 83 passed，
模型实际生成的补丁为 84 passed（83 个验证案例加 Agent 新增案例）。
汇总及文件 hash 见 `resource/manifests/hpc-smoke.json`。

完整检查顺序见 [本地评测适配](cluster-runtime.md)。至少保留真实任务的 prediction、
trace、补丁、测试日志，以及未修复失败 / gold 通过的验证证据。
模型没解决任务和评测环境出错必须分开记录。L4 单题结果及正式评分已取得；
更改 Agent、预算或 adapter 后，仍需重新验证新配置的运行时间。

另已准备并发起私有 `jwj1342/gemma4-official-runtime-smoke` 第 1 版，用官方 starter
的固定基础镜像、4×L4、关闭 Internet、最多 2700 秒，运行同一个 `rich_3454` 任务。
它核对 41 个官方 wheel 的 hash，按编译 → empty/gold → 推理 → agent 验证顺序执行。
该 notebook 已完成，并取回逐项日志：empty 为 1 failed / 82 passed，gold 为
83 passed，Agent 为 84 passed；官方编译及 41 个 wheel hash 检查通过。
真实设备为 4 张 NVIDIA L4，Agent 阶段 74.10 秒、9 次 LLM 调用、7 次工具调用；
该时间不含模型启动。notebook 从开始到 smoke 完成约 655 秒，不含排队与最终导出。
输出 ZIP 与正式提交的 SHA256 相同。证据见
[验证记录](../resource/manifests/kaggle-runtime-smoke.json)。
这是单题运行验证，不是一次新比赛提交。生成脚本为 `prepare_kaggle_runtime.py`。
Kaggle 使用官方 subprocess 后端且保留平台挂载的数据 Inputs，不能把它说成已经复现了
本地 Apptainer 的文件挂载隔离，亦不能凭单题结果宣布完整 benchmark 或预算验证完成。

查询或取得该验证 notebook 的结果（不会新建版本）：

```bash
bash -l scripts/kaggle.sh kernels status jwj1342/gemma4-official-runtime-smoke
# 状态进入 COMPLETE 或 ERROR 后保留输出，勿因排队而重新 push：
bash -l scripts/kaggle.sh kernels output jwj1342/gemma4-official-runtime-smoke \
  -p runs/kaggle-runtime-smoke/kaggle-output
```

`COMPLETE` 后仍须检查下载的测试日志和汇总；`ERROR` 时先检查错误发生在安装、
模型启动、推理还是测试阶段。Notebook 自身的状态不能代替正式提交的成绩。

## 3. Kaggle 私有打包 notebook

`scripts/prepare_kaggle_upload.py --help` 列出参数。脚本将已检查的小型 ZIP 嵌入 notebook，
该 notebook 只验证并输出 `/kaggle/working/submission.zip`，不用 GPU，也不需要 Internet。
它不是训练或推理 notebook；带有大 adapter 的包不能照搬此内嵌方式。

已实际完成的 v1：

```bash
bash -l scripts/kaggle.sh kernels push -p runs/pipeline-baseline/kaggle -t 600
bash -l scripts/kaggle.sh kernels status jwj1342/gemma4-pipeline-baseline
bash -l scripts/kaggle.sh kernels output jwj1342/gemma4-pipeline-baseline \
  -p runs/pipeline-baseline/kaggle-output
```

结果为 `COMPLETE`。下载 ZIP 与本地逐字节一致；详见
`resource/manifests/kaggle-pipeline.json`。页面：
<https://www.kaggle.com/code/jwj1342/gemma4-pipeline-baseline>（私有）。

## 4. 正式比赛提交

每天只有 1 次。先完成编译、真实任务验证与预算检查，再上传对应 `submission.zip`。
2026-10-05 API 明确返回 `is_kernels_submissions_only=false`、未关闭提交、账号已报名，
且尚余 1 次；因此本比赛支持直接上传 ZIP，不要求先创建打包 notebook。
证据为 `resource/manifests/submission-route.json`，执行前仍会复查。
首次上传已在 06:08 UTC 被接收：ID **56844353**，无 LoRA 基线、ZIP 3695 字节，
hash 与上述 v1 相同。接收回执见 `resource/manifests/competition-submission.json`。
最新状态 **COMPLETE**，公开分数 **0.12**，未返回 private 分数。结果见
`resource/manifests/competition-result.json`；上传前的“尚余 1 次”是历史值，不是当前剩余额度。

```bash
bash -l scripts/kaggle.sh competitions submit --help
bash -l scripts/kaggle.sh competitions submissions gemma-4-developer-agent -v
```

`scripts/submit_competition.py` 会重新核对真实 smoke 结果、冻结的配置、ZIP 与逐文件
SHA256，并检查当前比赛入口和剩余额度。默认只检查，加入 `--submit` 才实际上传：

```bash
module load StdEnv/2023 python/3.12.4
source /home/jwj/venvs/kaggle-cli/bin/activate
python scripts/submit_competition.py --run-dir runs/<已完成的实验>
python scripts/submit_competition.py --run-dir runs/<已完成的实验> --submit
# 查询现有提交和 L4 notebook，只读，不消耗提交次数：
python scripts/check_submission_status.py --submission-ref 56844353
```

上传前先以独占创建方式写 `competition-submission.json`，记录 ZIP hash 和请求状态；
已有记录时拒绝重复上传。网络异常或没有返回 submission ID 时，先查 Kaggle 历史，
不要删除记录后盲目重试。H100 单题验证证明本地链路；Kaggle L4 单题结果和正式公开分数已取得。当前每题 cap 为 4 分钟，约 120 题的理论生成预算为 8 小时，
剩余时间留给初始化等开销；这项估算不是 L4 全量实测。

提交后记录 ID、模型/数据/harness 版本、ZIP hash、状态、得分和错误日志。
仅查询已有提交，不因等待或错误状态自动重复上传。最终选择页面允许选择两个提交，
不要假定最后上传的一个自动就是最终候选。

官方依据：比赛 `Evaluation`、`Rules`、官方 Getting Started；原文见 `resource/competition/`。
