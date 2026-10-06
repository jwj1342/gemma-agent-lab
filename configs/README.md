# 配置

版本控制 agent YAML、数据切分规则、训练参数、环境版本和评测参数。
账户凭据放 HOME；不要将密钥写进配置。

`kaggle-cli.requirements.txt` 是本次下载工具环境快照，不是训练依赖。
评测依赖与训练依赖应分别锁定，具体约束见 [训练准备](../docs/training-plan.md)。
