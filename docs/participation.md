# 如何参与与协作

1. 打开[预测赛页面](https://www.kaggle.com/competitions/gemma-4-developer-agent)，登录自己的
   Kaggle 账号，阅读并接受官方规则，完成页面要求的验证。2026-10-04 API 快照确认
   `jwj1342` 已加入；新成员仍须自行完成自己的报名步骤。
2. 在比赛 Team 页面建立/加入正式团队，最多 5 人，合队截止 2026-11-25。
   仅 Linux 同组不等于 Kaggle 已组队。私下共享比赛代码/数据应遵守官方 team 规则。
3. 阅读 [Overview 导读](overview.md)、[Data 导读](data.md) 和完整官方规则快照。
   不将自己的 API token 分享给团队；每人使用各自 `~/.kaggle/access_token`，权限 0600。
4. 团队代码在 `src/`、配置在 `configs/`；原始资料留在 `resource/`，原始数据不改写。
   每次实验记录版本、切分、模型 revision、依赖、预算、随机种子和结果。
5. 在 Trillium 先读全局集群指南。代码编辑/提交在登录节点，下载/解压用 datamover，
   推理/训练由 Slurm 分配。PROJECT/HOME 在计算节点只读，运行产物用 SCRATCH。

## Kaggle 开发资源

从[官方 starter](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)
Copy & Edit 后确认 Inputs 包含比赛数据、官方 wheelhouse 和规定的模型版本。
在 Session options 选择 L4×4，Internet 关闭。不要给只负责 ZIP 打包的 notebook
申请 GPU；需要推理验证时才用 L4。

2026-10-05 原始 API 确认本账号 GPU 总额度为 30 小时；按 L4 的 2 倍倍率折合约 15 小时。
早期“6 小时”来自 SDK 序列化时漏掉一天的错误，已纠正，并非额度刚增加。
执行前用 `kaggle quota` 或原始 API 查看已用/预留量与实际剩余，不能依赖旧快照。
评分 12 小时与个人 Notebook quota 分开理解。

## 预测赛与论文赛

预测赛每天最多提交一次，因此正式上传前检查文件与验证证据，不循环试错上传。
最终可选两个 submission；以最终选择页面为准，不默认最后一次就是最终选择。
可选[论文赛](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)单独报名，
截止 2026-11-12；它不替代预测赛提交。

赛程、奖项与许可义务分别保存在 `resource/competition/timeline.md`、`prizes.md`、
`rules.md`、`foundational-rules.md`。不要擅自接受其他成员的法律条款或替他们报名。

## GitHub 协作

团队私有仓库为 [jwj1342/kaggle_gemma4_agent](https://github.com/jwj1342/kaggle_gemma4_agent)，
默认分支 `main`。每位成员使用自己的 GitHub 身份；维护者按成员 GitHub 用户名邀请。
完整克隆、轻量检查与 PR 流程见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

数据、权重、token、缓存、完整日志与检查点不进入 Git；实验结论和可复现配置进入 PR。
GitHub 保存已推送的源码和轻量资料，SCRATCH 中被忽略的大文件仍需单独备份。
Trillium 的文件 ACL、GitHub 仓库权限、Kaggle 正式队伍是三个独立的权限设置。
