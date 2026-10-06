# 维护脚本

轻量脚本先加载本机工具环境：

```bash
module load StdEnv/2023 python/3.12.4
source /home/jwj/venvs/kaggle-cli/bin/activate
```

| 脚本 | 用途 |
| --- | --- |
| `kaggle.sh` | 自动加载上述环境并调用 CLI；凭据从 HOME 读取 |
| `check_kaggle_quota.py` | 使用 `total_seconds()` 保存真实额度，避免 SDK to_dict() 漏掉整天 |
| `download_competition.sh` | 下载完整 ZIP，再按已有官方清单校验解包；在 datamover 执行 |
| `extract_competition.py` | 对照清单检查路径、名称、大小与 ZIP CRC；会覆盖目标同名原始文件 |
| `audit_data.py` | 遍历 tasks、graph、NPZ 头、wheel metadata、TGZ 成员；在 datamover 执行 |
| `collect_resources.py` | 获取官方页面、公开 Code、讨论回复树、官方排行榜 CSV；网络只读 |
| `collect_discussion_bodies.py` | 补齐讨论首帖、作者、完整回复；网络只读 |
| `index_resources.py` | 从本地快照生成导航与覆盖清单，不访问网络、不执行第三方源码 |
| `render_notebook_reading.py` | 从人工阅读记录生成逐项评语；检查源码 hash 和行号，不用关键词自动宣称阅读完成 |
| `download_eval_assets.sh` / `verify_eval_assets.py` | 下载并校验指定底座和完整官方 wheelhouse；datamover 执行 |
| `build_submission.py` | 确定性 ZIP、路径/扩展/容量检查与 CRC；不替代官方编译 |
| `prepare_kaggle_upload.py` | 生成私有 CPU 打包 notebook，不进行模型推理 |
| `prepare_kaggle_runtime.py` | 生成私有离线 L4×4 单题验证 notebook；与 CPU 打包 notebook 分开 |
| `prepare_eval_image.sh` / `prepare_eval_environment.sh` | 固定镜像与独立评测环境；已完成并通过单题实测 |
| `validate_agent.py` | 调用比赛官方 compiler 与限制；需要完整运行依赖 |
| `prepare_eval_task.py` | 仅保留推理输入字段，剔除参考补丁与验证测试 |
| `prepare_eval_run.py` | 冻结某次实验的 agent、公开任务与 bundle hash，不覆盖旧 run |
| `summarize_eval_smoke.py` | 检查 empty/gold 控制结果，分别记录流程执行与 agent 解题成功 |
| `run_eval_container.sh` / `evaluate_agent.py` | 分离推理和验证；评测须在计算 allocation 内 |
| `slurm_eval_smoke.sh` | 单 GPU 真实任务与 empty/gold/agent 验证入口；实际进度见本地评测适配文档 |
| `prepare_eval_git.sh` | 在 datamover 用签名 Ubuntu 包补齐本地容器缺失的 Git；保存包 hash 与 diff 实测 |
| `submit_competition.py` | 复查已测试 ZIP；默认只检查，`--submit` 上传并保存防重复的回执 |
| `check_submission_status.py` | 查询现有正式提交与 L4 notebook，保存状态快照，不重新上传 |

采集脚本使用缓存文件；更新已有条目时需明确选择新快照或替换哪些缓存，不能把再次运行
自动视为全部刷新。完整更新还需重新获取比赛文件清单、Models 页、wheelhouse 及 Code
结构扫描；这些不包含在单个采集命令中。保留旧的日期快照，先核对变更再更新研究报告。

以上是本账号路径配置。团队其他账号应使用自己的 HOME venv 和凭据，不共享 token。
