# HD1910 双足机器人训练代码

本仓库保存 Microduck 原始双足结构、HD1910 舵机适配，以及保留的行走和起身策略。
恢复基线为 `pollen-robotics/microduck_rl` 的
[`ceca01706143d729305baa94cbd8ab990979e325`](https://github.com/pollen-robotics/microduck_rl/commit/ceca01706143d729305baa94cbd8ab990979e325)。

- [恢复范围与验证结果](docs/HD1910_RESTORED.md)
- [HD1910 参数与运行方式](docs/hd1910-baseline.md)
- [策略来源与评估说明](models/SOURCE.md)
- [舵机参数来源及许可证](src/mjlab_microduck/robot/hd1910/README.md)

## 保留策略

| 用途 | ONNX | 训练检查点 |
| --- | --- | --- |
| 行走，第 4999 次迭代 | `models/hd1910_velocity_final.onnx` | `models/hd1910_walk_teacher.pt` |
| 起身，第 5999 次迭代 | `models/hd1910_standup_best.onnx` | `models/hd1910_stand_teacher.pt` |

两套策略均为 61 维观测、14 维动作，50 Hz；ONNX 包含观测归一化。
本仓库不包含本地训练日志、回退备份、Python 虚拟环境和后续 2804 主动轮实验。
恢复记录中提及的本地备份／完整日志路径不会随源码上传。

## 运行

需要 Python 3.12、uv 和支持的 CUDA GPU。

```powershell
uv sync --frozen
powershell -ExecutionPolicy Bypass -File scripts/view_hd1910.ps1
uv run scripts/view_standup_push.py
uv run scripts/view_walk_recovery.py
```

训练入口及更多说明见 [HD1910 基线文档](docs/hd1910-baseline.md) 和
[上游 README](README.md)。策略的成功率来自有限仿真测试，没有实机验证。

## 许可证

项目来源和原始版权声明保留：源码遵循 [Apache-2.0](LICENSE)，模型网格遵循
[模型许可证](LICENSE-MODELS.md)，舵机参数保留其来源许可证。
仓库最初初始化时的 MIT 许可证另存为 [LICENSE.repository-MIT](LICENSE.repository-MIT)；
它不改变上游源码和模型资产原有的许可条件。
