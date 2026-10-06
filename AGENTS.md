# 项目协作指南

- 本项目参加 `gemma-4-developer-agent`；paper track 是另一项比赛。
- 仅在 Trillium 操作时先读 `/home/jwj/.codex/cluster-guides/trillium.md`；其他电脑不依赖此路径。项目说明见
  `README.md`、`docs/competition-research.md` 和 `docs/training-plan.md`。
- 代码、配置、实验说明进入 Git。数据、权重、运行缓存、检查点和凭据按 `.gitignore`
  排除；原始比赛文件与上游 notebook 保持不修改，实验改动写入 `src/`、`configs/`。
- 新实验使用 `runs/<日期>-<简短名称>/`，记录 Git revision、模块、依赖、数据版本、
  模型 revision、随机种子、资源、指标和完整恢复路径。
- 不在登录节点运行评测、训练或批量解包仓库；大传输在 datamover，计算由 Slurm 分配。
- 资料是日期快照。优先级为当前比赛规则/主办方明确回复、当前官方包行为、公开案例；
  区分作者宣称成绩、静态代码分析和独立运行验证。
- 正式评测底座目前固定 `gemma-4-31b-it-qat-w4a16-ct`；Models 页其他模型不表示获准。
- 评测环境与训练环境分开；第三方 notebook 只能作为资料，不直接执行其安装/删除单元。
- HOME 中 Kaggle token 不复制进工作区。训练作业无需也不应携带 Kaggle 凭据。
- 团队成员先加入正式 Kaggle team；共享权限按成员需求设置，不全局开放 SCRATCH。
- GitHub 仓库为 `jwj1342/kaggle_gemma4_agent`（私有）；新成员流程见 `CONTRIBUTING.md`。
  不将本机已下载的数据/模型、HOME 环境或历史 run 路径视为 clone 后自动可用。
- Kaggle quota 用 `scripts/check_kaggle_quota.py` 或 CLI 查询；kagglesdk 0.1.37 的
  `to_dict()` 会漏掉 duration 的整天。早期保存的 6 小时值有误，原始 API 已确认总额度
  108000 秒（30 小时）；使用 `total_seconds()`，运行前再查已用与预留。
- 每次评测先用 `prepare_eval_run.py` 固定 agent/公开任务/ZIP hash；运行器读取 run
  内配置快照。empty/gold 控制实验通过后才跑推理，分别记录流程执行与解题成功。
