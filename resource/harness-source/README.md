# 官方 harness 源码快照

来自比赛官方 [wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)。
仅解包 `.py` 与 wheel `METADATA`，未导入、安装或执行。
三个 wheel 的版本、字节数和 SHA-256 见 [清单](../manifests/harness-wheels.json)。

本次重点核对：

- `swegemma/config.py`：3 GiB 限制、扩展白名单、输出/思考 token 范围。
- `adk_submission/server.py`：根据实际 adapter 数量与 rank 设置 vLLM LoRA 参数。
- `adk_submission/resolvers/generation.py`：thinking budget、seed 到 LiteLLM/vLLM 的转发。
- `swegemma/tools/graph.py`：数字参数转换、查询结果截断。
- `swegemma/graph/`：现有图/embedding 的检索路径；不能假定支持任意自然语言 embedding。

源码检查不能代替依赖安装、模型服务启动或评分环境实测。
