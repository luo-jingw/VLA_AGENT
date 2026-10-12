# 交付

## 目录

```text
jax/params/                         JAX (orbax) 权重；LoRA 训练时保留独立的适配器
jax/assets/<asset_id>/              norm_stats.json, arc_resample.json
pytorch/model.safetensors           PyTorch 权重（bf16，已合并 LoRA），通过一致性检查后才放入
pytorch/config.json
pytorch/assets/<asset_id>/          同上
deploy/arc_interp.py                插值，与训练标签同一版本
deploy/<task>_serve_config.py       推理配置，只依赖上游 openpi 与 arc_interp.py
README.md
```

`train_state/` 只在续训时需要，不放进交付目录。`jax/` 和 `pytorch/` 各自都是完整的 openpi checkpoint 目录。

## README 的内容

1. 模型是什么：基座、微调方式、数据（来源、集数、帧数、fps、相机、动作维）、任务；
2. 醒目提示：每次输出的点是否逐帧；不是逐帧时必须先插值，并说明不插值的后果；
3. 文件表；
4. 输入输出：观测键与格式、状态/动作布局、单位、指令维的取值含义；
5. 在已有服务里使用：推理配置的必需项、插值的一行调用、建议的执行长度；
6. 标签是怎么构建的、服务端假定了什么；
7. 训练超参；
8. 评估：留出集与开环结果，注明不是任务成功率；做过闭环试验时给出成功率与置信区间；
9. 局限：数据覆盖、录制性质（例如录制在任务完成时停止）、放置误差、PyTorch 一致性检查结果；
10. 许可：基座权重的条款随之适用。

写 README 时不出现内部机器名和本地路径，只写仓库内的相对路径。

## 上传

- 上传前确认目标仓库的可见性（公开还是私有），并在交付说明中写明；
- token 使用项目范围内的凭据，来源记在 `PROJECT.md`；不打印，也不写入任何文件；
- 小文件（README、deploy/）先上传，大文件后上传；
- 上传完成后列出远端文件，逐个比对大小；对 safetensors 这类单个大文件再比对校验和。
