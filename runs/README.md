# 实验产物

每次实验用独立的 `<日期>-<简短名称>/` 子目录，记录 Git revision、模块/依赖版本、
模型与数据版本、随机种子、资源、指标、日志和恢复路径。产物不进入 Git。
当前已有资料采集、打包、容器准备与实际评测记录，没有训练结果。

- `hpc-smoke-20261005-r4/`：成功的单题离线评测；官方编译、empty/gold 对照、真实
  模型补丁和独立测试全部执行。旧的三个 run 保留排障证据。
- `pipeline-baseline/`：确定性 ZIP 与已完成的 Kaggle CPU 打包 notebook 输出。
- `kaggle-runtime-smoke/`：已提交的私有 L4×4 验证 notebook；结果按远程状态判断。
- `slides-preview/`：slides 编译日志与渲染检查图。

汇总记录另存 `resource/manifests/hpc-smoke.json` 和 `kaggle-evaluation-status.json`。
正式比赛上传回执在成功 run 中；不要复制新目录后据此重复上传。

计算节点的可写缓存也放 SCRATCH。重要源码和最终成果需另行备份；本目录不备份。
