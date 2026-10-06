# 数据

`raw/gemma-4-developer-agent/` 保存官方原始包，524 个文件、20.88 GiB，不直接修改。
`processed/` 保存有明确转换脚本和版本清单的派生数据；二者均不进 Git。

原始 `tasks.jsonl` 含 gold patch/test patch。推理输入与评分标签应分开管理；
训练/验证按 repo 与 base commit 分组。仓库 TGZ 按所需任务展开到各 run，
不在登录节点批量解压或执行仓库测试。

完整清点见 [数据审计](../resource/manifests/data-audit.json) 与
[比赛说明](../docs/competition-research.md)。
