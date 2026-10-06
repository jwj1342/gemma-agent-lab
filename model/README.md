# 模型与 adapters

- `base/`：固定版本的底座、tokenizer、配置和模型清单。
- `adapters/`：通过正式评测底座加载验证的导出 adapter。
- 训练中间 checkpoint 放 `runs/<run-id>/`，不和最终 adapter 混放。

指定底座 v2 已下载并完成文件大小、SHA256 与 safetensors offset 边界检查；尚未训练 adapter。
离线模型加载和单题补丁生成已通过，记录见 `resource/manifests/hpc-smoke.json`；文件
校验见 `resource/manifests/eval-assets-verification.json`。模型卡与版本记录见
[Models 资料](../resource/models/README.md)。大文件不进入 Git；每份权重另存来源、
revision、hash、许可和导出环境。不同实验优先引用同一只读底座，避免重复占空间。
