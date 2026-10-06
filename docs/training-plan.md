# Trillium 上的训练与评测准备

本文件是后续实现依据，不是已经跑通训练的声明。已安装的 `/home/jwj/venvs/kaggle-cli`
仅用于数据获取。最新 PyTorch / SFT / RL 操作知识已并入
`/home/jwj/.codex/cluster-guides/trillium.md`，用户级 AGENTS 会提醒后续会话先读。

## 两套环境

1. **比赛评测环境**：复现官方 wheelhouse、定制 vLLM、ADK、swegemma、模型版本及
   chat template；先离线完成一题生成与验证。不要用 Alliance 最新 vLLM 替换它后声称
   与评分环境等价。官方 Notebook 安装单元假定 Kaggle 基础镜像已预装很多依赖。
2. **训练环境**：Trillium modules + wheelhouse + HOME venv；Torch、Transformers、PEFT、
   TRL、量化/attention 扩展版本按目标模型组合验证。训练结束的 adapter 必须回到第一套
   环境加载，检查目标层/形状/量化底座兼容性、实际生效及 3 GiB 提交限额。

Dockerfile 并不意味着 Trillium 用户有 Docker daemon。可研究 Apptainer 或受控 subprocess
后端，但后者不是 Docker/gVisor 的等价安全边界。模型产生的命令必须限制在专门工作区，
环境中不携带下载凭据；不要让 agent 修改 HOME/项目源码。Apptainer 内 subprocess
方案已通过挂载、可写路径、官方编译、离线模型加载和 empty/gold 对照测试。
完整补丁生成／独立验证已在公开题 `rich_3454` 上通过，详见
[运行适配状态](cluster-runtime.md)。这验证了单题评测流程，尚未验证训练。

## 建议顺序

1. 固定公开题切分与环境，验证选定题无补丁失败、有 gold patch 成功，排除不可解或环境
   不完整的案例。按 repo/base commit 分组，避免同一快照进入训练和验证两侧。
2. 准备正式 v2 底座和 tokenizer/config 文件，数据与 caches 放 SCRATCH；单张 H100 短作业
   做推理、harness 两阶段评估，保存每题 patch、trace、用时和失败类型。
3. 建立无 adapter 的 analyzer/coder baseline；在接近 L4 的评测环境复核预算。
4. SFT 从验证成功的工具调用轨迹开始；检查 chat template、tool schema、loss mask，
   小量训练步后保存、恢复、导出、重载。不要把 issue→gold diff 训练自动当作完整 agent 训练。
5. 有可靠偏好对再研究 DPO；有稳定、可承担的生成与沙箱测试成本再研究 GRPO。
   奖励分别记录代码通过、环境错误、无补丁、格式错误、超时，避免训练去修改测试环境。

官方资料入口与要核对的内容：

| 资料 | 当前用途 |
| --- | --- |
| [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) | messages/工具轨迹、assistant loss mask、packing |
| [PEFT 量化](https://huggingface.co/docs/peft/en/developer_guides/quantization) | 量化训练准备和 LoRA 目标模块；W4A16 不等于 NF4 |
| [TRL DPO](https://huggingface.co/docs/trl/en/dpo_trainer) | chosen/rejected 对、参考模型与显存预算 |
| [TRL GRPO](https://huggingface.co/docs/trl/en/grpo_trainer) | 多候选、工具环境和自定义 reward；需要 SWE 环境接入 |
| [TRL vLLM](https://huggingface.co/docs/trl/en/vllm_integration) | server/colocate、权重同步、版本兼容 |
| [PyTorch torchrun](https://docs.pytorch.org/docs/main/elastic/run.html) | rank、进程层级与 Slurm 集成 |
| [Lightning checkpoint](https://lightning.ai/docs/pytorch/2.5.0/common/checkpointing_basic) | `fit(..., ckpt_path=...)` 恢复完整状态 |

最新 TRL 文档列出的 vLLM 支持区间为 0.20–0.30，而比赛 wheelhouse 是定制 0.19.1。
这是环境应隔离的具体原因；实现时再对照安装版本，而不是强行把两个依赖集合装在一起。

## 本账号资源与存储

同时看容量和文件数量两个限额。2026-10-05 的 `diskusage_report` 与当前 `df` 上限相差
约 10%；10-04 早期报告也曾显示较大的值。尚未核实这是工具口径差异还是软/硬阈值，
不能据此宣布配额增加或减少。日常预算采用当前 `diskusage_report` 的较小值。
其他系统的 `df` 可能显示整个文件系统，不能将它普遍当成个人配额工具。

| 存储 | 当前报告额度 / 文件数 | 下载模型前的用量快照（取整） | 用法 |
| --- | --- | --- | --- |
| HOME | 100 GiB / 100 万 | 约 1.42 GiB / 1.4 万 | 环境、配置；计算节点只读 |
| SCRATCH | 25 TiB / 1,000 万 | 约 41 GiB / 2,731 | 数据、权重、缓存、checkpoint；无备份 |
| PROJECT def-zhouyang | 1 TiB / 200 万，组共享 | 报告 0 B / 4 | 重要源码/成果；计算节点只读 |

同一时段 `df` 上限：HOME 110 GiB / 110 万文件；SCRATCH 27.5 TiB / 1,100 万；
PROJECT 约 1126.4 GiB / 220 万。额外约 10% 不纳入常规实验预算。
25 TiB 是磁盘空间，不能与 CPU RAM 或 GPU VRAM 相加。上表采集后已下载模型和容器，
因此 41 GiB 不再代表当前用量；运行 `diskusage_report` 查看新的统计，留意更新延迟。

本项目原始包和 ZIP 共约 41.30 GiB，仅占 SCRATCH 很小部分。但如果展开所有仓库，
会再增加约 21.13 GiB 和 23.8 万个条目。模型 v2 包约 21.70 GiB，训练底座、转换副本、
optimizer/checkpoint、多个实验缓存还要另外预算；磁盘上的权重大小不等于训练显存需求。
2026-10-05 安装评测环境后的报告为 HOME 6094 MiB / 96K 文件、SCRATCH 81 GiB /
4885 文件、PROJECT 0 B / 4 文件。原始快照在
`resource/manifests/diskusage-2026-10-05-runtime.txt`；构建临时文件和缓存会改变用量，
报告也可能有更新延迟，不把旧用量或短期余量当成固定预算。

计算节点无外网，HOME/PROJECT 只读；RAM disk 占内存。运行前配置 HF/Torch/Triton/XDG
缓存、日志、checkpoint 到各 run 的 SCRATCH 路径，关闭在线跟踪。不要将 credential token
放在团队缓存中。每次核实 GPU 状态与 account；当前已验证默认 account 为 `def-zhouyang`。

## 开跑前仍缺少的具体项

- 指定底座 v2 和完整官方 wheelhouse 已下载，已在实际 H100 作业内离线加载；
  vLLM 报告权重加载占 19.62 GiB，这不是包含 KV cache 等在内的峰值显存。
- 评测容器已通过官方编译、empty/gold 对照和真实补丁独立测试。曾遇到的镜像缺 Git
  问题已修复；作业 1033882 正常完成，Agent 补丁 84 项测试通过。正式提交 56844353 已完成，公开分数 0.12；L4×4 单题验证也已通过。
- 独立训练环境尚未建立；不能将预检查环境视为完整训练或评测环境。
- 已有单卡 smoke 作业及真实生成吞吐日志；尚未验证训练、检查点保存/恢复和多卡训练。
- 团队尚未确定 baseline、验证切分、adapter 架构和训练预算。

下一步建立分组验证集，再决定 SFT/RL 与多卡规模。Kaggle L4 单题 agent 阶段已测得 74.10 秒（不含模型启动），这不等于训练已跑通。

## 公开训练示例对我们的具体启示

全部 Code 的逐项记录见 [阅读表](../resource/notebooks/reading-notes.md)。本次阅读后，
训练路线仍优先放在可靠评测之后：

- 31B QLoRA study 有真实训练入口与 prompt loss mask，但训练上下文使用 gold hunk 定位，
  且 bitsandbytes 4bit 底座与正式 QAT W4A16 服务不同。应分别测“已知位置补丁生成”和
  “未知位置完整 agent”，不能合并为一个泛化结论。
- issue→diff 示例通常缺少源码观察和工具执行结果；不能仅因 loss 下降就推断多步 Agent
  改善。仓库级 gold 文件先验也只能从训练侧估计，不能用全部公开题先验做同集评估。
- 从 Gemma 2 导出并改写模型名称，或用随机 A/全零 B 制作 safetensors，不能替代目标
  Gemma 4 上的训练和服务重载。应检查实际 trainable tensors、梯度、保存/恢复，以及
  adapter 开关前后在固定输入上的差异。
- 先做少量步骤和明确的独立验证，再扩大 SFT；暂不因社区标题或占位权重开始大规模 RL。
  评测依赖失败、跳过必需测试、修改测试环境等情况须单独记录，不能变成奖励捷径。
