# Trillium 本地评测适配状态

本页只记录本项目实现。操作前仍须阅读 `/home/jwj/.codex/cluster-guides/trillium.md`。
模型推理和仓库测试不得直接在登录节点执行。

## 已准备与仍待验证

- 已下载指定模型 v2（8 文件）和 41 个官方评测 wheelhouse 文件；文件大小与官方清单
  相符，SHA256、全部 wheel CRC、safetensors offset 边界已检查。
  证据为 `resource/manifests/eval-assets-verification.json`；这些检查不执行模型。
- 已编写 Apptainer 构建、HOME 环境准备、隔离运行和 Slurm 单卡 smoke 脚本。
- 镜像已构建成功并通过 CPU Torch 导入检查，使用官方 `vllm/vllm-openai:v0.19.1` 的固定 amd64 digest，详见
  `resource/manifests/vllm-image.json`；构建日志为 `runs/prepare-eval-image.log`。
- native `gemma4-compiler` 环境仅通过导入、目录规则及指定模型检查，不等同完整编译。
  需先加载 `StdEnv/2023 python/3.12.4 arrow/25.0.0`；其 Google GenAI 等依赖不代表
  Kaggle 的精确运行组合，只用于预检查。
- HOME 中的容器 Python 环境已补齐依赖并保存版本锁；只剩下文列出的已知 metadata
  冲突。完整官方 compiler、单题 GPU 推理和独立补丁测试已通过；实测见本页末尾。

## 准备步骤

以下下载和环境构建在 `trig-dm1` 上进行，不占用登录节点进行大传输或容器打包。
已有运行中的构建不要重复启动。

```bash
ssh -o BatchMode=yes trig-dm1 \
  'bash -l /scratch/jwj/kaggle_gemma4_agent/scripts/download_eval_assets.sh'
ssh -o BatchMode=yes trig-dm1 \
  'bash -l /scratch/jwj/kaggle_gemma4_agent/scripts/prepare_eval_image.sh'
# 镜像成功后再安装官方 wheels 与缺失依赖：
ssh -o BatchMode=yes trig-dm1 \
  'bash -l /scratch/jwj/kaggle_gemma4_agent/scripts/prepare_eval_environment.sh'
ssh -o BatchMode=yes trig-dm1 \
  'bash -l /scratch/jwj/kaggle_gemma4_agent/scripts/prepare_eval_git.sh'
```

官方 wheel 的文件名可能丢失 CUDA local-version 后缀；安装脚本读取 wheel 内部 metadata，
以准确版本创建临时文件名，保留原 wheel 不修改。仅当安装来源记录的 wheel SHA256
与官方文件完全一致时跳过重装；不能因版本号相同跳过赛事补丁。实际执行仍须检查
Torch / CUDA / vLLM 兼容性。

### 官方 wheelhouse 的依赖冲突

2026-10-05 实际安装发现，不能把全部官方 wheel 交给 pip 一次性解析：

- `google-adk 1.36.1` 要求 OpenTelemetry SDK ≤1.41.1，但官方另附的 OTLP
  metapackage 1.44.0 要求 1.44 系列。当前保留基础镜像的 1.41.0 运行组件，关闭遥测。
- 官方 `prometheus-fastapi-instrumentator 8.1.0` 声明 Starlette ≥1，而 ADK 要求 <1。
- 官方 `xgrammar 0.2.5.post1` 声明 Transformers <5，官方同时提供 Transformers 5.13.1。

官方 starter 使用 `--no-deps` 安装。当前本地适配同样保留 41 个官方 wheel 的原始
内容；单独解析缺失的 ADK、LiteLLM 和测试依赖，并固定官方包版本，避免被 PyPI
同名版本替换。`quack-kernels` 所需的 Cutlass DSL 补到其声明的 4.6.2。
`pip check` 的原始结果与逐条原因记录在
`configs/eval-container.requirements.dependency-audit.json`；任何未列明的新冲突都会
使准备失败。基础镜像另有与本任务无关的 PyGObject/pycairo 缺失项，单独记录。
这些例外**不表示依赖检查完全通过**。完整编译、empty/gold 对照和单题 GPU 推理
已通过，但只能证明本次执行涉及的路径；不能推广到全部扩展、训练或其他任务。
Kaggle 验证保持官方镜像和安装方式。

已从展开后的镜像静态确认 Torch 2.10.0+cu129 / CUDA 12.9；GPU 登录节点当前 driver
为 580.173.02。实际 H100 作业已完成模型加载与推理。基镜像没有 pytest，而官方 subprocess
backend 跳过 Docker 的逐题依赖安装，继承已有 Python 环境。因此准备阶段另外从比赛
测试 wheelhouse 安装 pytest 8.3.4，并安装 Dockerfile 指定的 pytest-timeout 2.1.0。
这是本地适配的测试工具版本，不能据此断言与私有评分镜像所有依赖完全相同。

镜像仅提供 `python3` / `python3.12`，没有 `python` 命令别名。现已实际执行
`/usr/bin/python3.12` 并导入 Torch，确认 Python 3.12.13 / Torch 2.10.0+cu129。
首次构建脚本末尾的检查因此返回非零，但 SIF 构建本身已成功；已修正入口并单独复验，
无需重新构建。证据见 `runs/inspect-eval-image-python3.log` 和 image manifest。

基础镜像也没有 Git。实际推理曾成功修改代码，却因 `submit_patch` 调用不到 `git`
而无法生成补丁，随后超时（作业 1033766）。现通过 Ubuntu 签名的软件源下载 Git 和
PCRE2 的 `.deb`，无 root 解包到 HOME venv 的 `vendor/git/`，由 venv 中的 wrapper
提供命令。`resource/manifests/eval-git.json` 保存版本、包 SHA256 与 init/add/commit/diff
实测结果。运行入口会在模型加载前检查必需命令和 Git 补丁生成，避免重复此失败。

## 隔离与写路径

运行器使用 `--containall --no-home`，禁用默认宿主文件系统、管理员 bind 与 cwd 挂载，
仅显式挂载所需目录。模型、Agent 配置、脚本、HOME 预建 venv、数据均只读。
推理侧只挂载 `prepare_eval_task.py` 生成的公开字段；验证侧另挂载完整任务标签。
每个阶段仅可写自身 run 子目录，HF、Torch、Triton、XDG、临时文件全部在 SCRATCH。
推理环境不挂载 Kaggle token，也不挂载全量项目或验证结果。

实测发现站点还导出 `APPTAINER_BIND=/project,/scratch`；它属于用户级绑定，
`--cleanenv --no-mount bind-paths` 不会清除。运行器现先清除 Apptainer/Singularity
的 bind/mount 环境变量，再传显式白名单。2026-10-05 复验全部通过：Agent、脚本和
venv 只读；Kaggle 凭据目录和完整项目不可见；run、tmp 可写。证据：
`runs/hpc-smoke-20261005/compile/runtime-filesystem.json`。

这里是 Apptainer 内的官方 subprocess backend，不是原生 Docker/gVisor；不能把容器
目录隔离等同评分服务的全部安全属性。挂载、GPU 模型加载和单题真实测试均已实测。

## 复现单题 smoke 验证

首先创建一个新的 run，冻结配置、构建对应 ZIP，并剔除推理输入中的验证标签：

```bash
module load StdEnv/2023 python/3.12.4
gemma_root=$(pwd -P)
gemma_run="$gemma_root/runs/$(date -u +%Y%m%dT%H%M%SZ)-rich3454"
python scripts/prepare_eval_run.py --run-dir "$gemma_run" \
  --agent-dir configs/baseline --task-id rich_3454
```

run 必须是新目录；若已经存在，先查看原有证据，后续重跑使用新名称。运行器挂载
`runs/<实验名>/inputs/agent/` 的配置快照，不随 `configs/baseline/` 的后续修改改变。
目前 `prepare_eval_run.py` 只负责复制和记录，不执行模型或测试。

从 GPU 登录节点使用 `scripts/slurm_eval_smoke.sh` 提交单 GPU 作业，明确 account，
日志与工作目录放 SCRATCH。普通 GPU 作业不显式指定 compute partition。

本账号在仓库根目录运行；沿用上一步同一 shell 中的两个变量：

```bash
sbatch --parsable --account=def-zhouyang \
  --output="$gemma_run/slurm-%j.out" --error="$gemma_run/slurm-%j.err" \
  "$gemma_root/scripts/slurm_eval_smoke.sh" "$gemma_root" "$gemma_run" rich_3454
```

命令返回 job ID，记录它再用 `squeue -j JOBID` 查询；结束后用
`sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,NodeList` 查看退出状态。
本次已成功的作业另外使用 `debug` 分区；上面给的是普通 GPU 队列入口，不继承临时
排障用的节点排除规则。不得因暂时没看到日志就重复提交。

```bash
python scripts/summarize_eval_smoke.py --run-dir "$gemma_run"
```

只有作业实际运行完再执行最终汇总。`pipeline_executed=true` 表示所有阶段确实执行，
`agent_resolved=true` 才表示这个补丁解决了该题；二者分开解释。
若失败，先查 `slurm-*.err`、`infer/results/failure.json`、`infer/results/vllm.log`
和相应阶段的 `tests.log`。环境错误修复后创建新 run，保留失败 run 的证据。

团队成员换账号时，先核实自己的 account、配额、模块和权限。当前脚本仍有本账号
HOME venv 路径，主要位于 `prepare_eval_environment.sh`、`prepare_eval_git.sh`、
`run_eval_container.sh`、`container_entry.sh`；复制仓库不会自动复制 HOME 环境、
被忽略的数据或凭据。先调整这些路径并完成准备步骤，再申请实际计算资源。

脚本依次进行官方编译、empty/gold 控制实验、推理和 agent 补丁验证。
`summarize_eval_smoke.py --controls-only` 先检查 empty 有真实测试失败、gold 通过，
通过后才启动模型；最后记录 `pipeline_executed` 与 `agent_resolved` 两个独立判断。
验证结果应证明：
未修复版本触发相关失败，gold 能通过，agent 的结果被真实测试解释。
保存依赖锁、版本、显存/时间、trace 与补丁；未拿到这些证据前，不宣布 pipeline 跑通。

## 2026-10-05 实测结果

`runs/hpc-smoke-20261005-r4`，作业 **1033882** 在 `trig0001` 使用 1 张 H100、
24 CPU 核，`COMPLETED / 0:0`，总用时 **5 分 49 秒**。

| 检查 | 实际结果 |
| --- | --- |
| 官方 compiler / 隔离与 Git 预检查 | 通过 |
| 原始代码 + 验证测试 | 1 failed / 82 passed，正确触发 URL 高亮问题 |
| gold 补丁 + 验证测试 | 83 passed |
| 实际模型推理 | 19 次 LLM 调用、12 次工具调用，生成 1161 字节补丁；83.85 秒含任务准备 |
| Agent 补丁 + 独立验证 | 84 passed，含 Agent 新增的一项测试 |

补丁为 URL 正则增加 `@`，并新增回归测试，没有删除或跳过原有测试。工具轨迹曾出现
4 次缺少 `old_string` 的编辑调用，模型随后恢复；可作为改进工具调用格式的观察，
不能据单个案例推断整个基线的失败分布。vLLM 权重加载报告 19.62 GiB，**不是峰值显存**。
TP=1、BF16、32768 context；没有训练、LoRA、保存/恢复或多卡训练验证。

汇总：`resource/manifests/hpc-smoke.json`。原始证据包括 run 内 `compile/validation.json`、
`infer/prediction.json`、`infer/results/traces/`、`infer/results/vllm.log`、
`verify/{empty,gold,agent}/tests.log` 与 `smoke-summary.json`。
正式上传 **56844353** 已被 Kaggle 接收；单题通过不等于完整 benchmark、L4 时限通过或
正式比赛得分。

排障历史保留，未覆盖旧 run：

| 作业 | 原因与处理 |
| --- | --- |
| 1033745 | 节点 prolog 报 GPU 无响应，系统取消；不是模型或脚本结果 |
| 1033752 | 预检查发现继承 bind 暴露全量 SCRATCH；清除专用运行器的继承 bind 后修复 |
| 1033766 | 模型已改代码但基础镜像缺 Git，无法导出补丁；补齐 Git 并增加补丁预检查 |
| 1033882 | 修复后完整单题链路通过 |
