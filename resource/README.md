# 比赛资源索引

采集快照：2026-10-04 晚至 10-05（UTC）。原始资料与我们的判断分开保存。
请先看 [比赛调研](../docs/competition-research.md)；数据可用性与阅读范围见
[覆盖清单](manifests/research-collection.json)。

| 目录 | 内容与覆盖范围 |
| --- | --- |
| [competition/](competition/pages.json) | 10 份完整官方页面，包括 Overview、Data description、规则、赛程和 L4 说明 |
| [notebooks/](notebooks/README.md) | 112 个公开条目全部下载，1 个源码为空；结构审阅与重点代码检查 |
| [Notebook 阅读表](notebooks/reading-notes.md) | 按条目记录实现、可借鉴部分、复现边界和源码位置；补读进度在表头 |
| [notebooks/official/](notebooks/official/getting-started-gemma-4-developer-agent.ipynb) | 用户指定的官方 starter，已审读 |
| [discussions/](discussions/README.md) | 116 个主题、370 条首帖及回复，作者和时间随原始 JSON 保留 |
| [models/](models/README.md) | Models 页全部 4 个实例及模型卡；指定 Gemma v2 已另行下载并校验 |
| [leaderboard/](leaderboard/README.md) | 官方完整 CSV、统计与 1,755 队分布 |
| [manifests/](manifests/data-audit.json) | 比赛文件清单、下载校验、数据结构检查和存储快照 |
| [harness-source/](harness-source/README.md) | 三个官方 wheel 的 Python 源码与 metadata，只作静态检查 |
| `downloads/` | 原始比赛 ZIP、下载缓存与小型 harness wheels；不进 Git |

第三方文件保持原样。不直接运行 notebook 安装/清理单元，不把作者自报成绩当作复现。
尚未全量检查外部链接、图片附件或每行第三方代码；没有执行第三方 notebook 的训练
或评测。团队实测见 [H100 运行记录](manifests/hpc-smoke.json)、
[Kaggle L4 验证](manifests/kaggle-runtime-smoke.json) 和
[正式评分结果](manifests/competition-result.json)。
公开资源可能更新，此目录是带来源的快照，不表示永久最新。

更新基础页面/Code/回复树：运行 `scripts/collect_resources.py`；补齐讨论首帖与作者：
运行 `scripts/collect_discussion_bodies.py`；完成新一轮核查后运行
`scripts/index_resources.py` 重建索引。Models 页与 wheelhouse 清单当前为本次独立 API
采集结果，基础采集脚本不会自动更新它们。工具均使用 HOME 的 kaggle-cli 环境。
