# HD1910 双足基线恢复记录

恢复日期：2026-11-10。目标：原始双足源码 + HD1910 舵机适配 + 已训练的行走、起身策略。

当前目录原本没有 `.git`，无法直接 checkout 历史工作区；本次从固定上游源码重建，
不是声称找到了某个完整的历史提交。备份保存在
`../.rollback_backup/20261110_biped_hd1910/`。

## 源码基线

采用下载目录的 `microduck_rl-develop.zip`，其 GitHub ZIP 注释指向
[`ceca01706143d729305baa94cbd8ab990979e325`](https://github.com/pollen-robotics/microduck_rl/commit/ceca01706143d729305baa94cbd8ab990979e325)。
已在线核对该提交与任务注册源码。没有用最新 develop 覆盖历史训练条件。

原始 `mdp.py`、双脚 XML、默认站姿、基础行走／起身配置、奖励、观测与随机化均恢复或
保持与该压缩包一致。所有原始文件均存在。

相对该上游基线保留的共享文件差异：

| 文件 | 差异原因 |
| --- | --- |
| `src/mjlab_microduck/tasks/__init__.py` | 增加 HD1910 Velocity、StandUp、VelStand 的 Flat/Rough 注册 |
| `pyproject.toml`、`uv.lock` | Windows CUDA torch/torchvision 路由，确保能训练 |
| `scripts/infer_policy.py` | Windows 终端输入和无 stdin 支持 |
| `tests/test_aarch64_cuda_torch.py` | 检查 Windows CUDA 轮子依赖配置 |
| `tests/test_hf_jobs_flag.py` | 支持检查 Windows PE 形式的 train 启动器 |
| `.gitignore` | 排除本次回退备份 |

额外源码仅保留 HD1910 执行器适配、参数与许可证、环境变体、训练／评估／回放脚本，
以及对应配置和 GPU 回归测试。后续 2804 模型、轮子控制器、轮子策略、专项侧移、
高度、坐站模仿和头部实验均已从活动源码／模型／日志目录移出。
上游原有被动 rollers 任务属于原始源码，予以保留。

## 保留策略

| 用途 | ONNX | 检查点副本 |
| --- | --- | --- |
| 行走，iteration 4999 | `models/hd1910_velocity_final.onnx` | `models/hd1910_walk_teacher.pt` |
| 起身，iteration 5999 | `models/hd1910_standup_best.onnx` | `models/hd1910_stand_teacher.pt` |

选择依据和 SHA-256 见 [models/SOURCE.md](../models/SOURCE.md) 与
`models/hd1910_policies.json`。多个起身检查点在本批评估中并列通过，因此选取最终
5999；不声称有限仿真能证明全局最优。

原始训练日志和检查点保留在 `logs/rsl_rl/microduck_hd1910_velocity/` 与
`logs/rsl_rl/microduck_hd1910_stand_teacher/`。本次验证报告在
`logs/hd1910_baseline_validation/`。

## 验证结果

- HD1910 配置及 GPU 物理回归：20 项通过。
- 其余源码回归：272 项通过、1 项跳过；1 项原始发布测试要求存在 Git checkout 的
  commit 信息，此 ZIP 工作区无 `.git`，因此该项失败。未为了通过测试伪造 Git 历史。
- 行走：4750／4999 各 56 个十秒回合均未终止，4999 归一化速度跟踪误差更低。
- 起身：5999 在四种初始状态各 16 个十五秒回合中达到并保持稳定双脚支撑。
- 两个 ONNX 均为 `[1,61] -> [1,14]`，检查图结构、有限输出与检查点数值一致性；
  32 个随机输入的最大误差分别为 7.63e-6、4.29e-6，包含归一化。
- 双策略执行 1100 步回放，施加一次随机推扰，并显式重置为俯卧：检测到
  WALK → RECOVERY → WALK，最终倾角 0.97 度、躯干高度 0.114 m。

仿真通过不等于实机验证。起身试验的精确成功条件见策略来源说明与 JSON 报告。

## 使用

```powershell
powershell -ExecutionPolicy Bypass -File scripts/view_hd1910.ps1
.venv/Scripts/python.exe scripts/view_standup_push.py
.venv/Scripts/python.exe scripts/view_walk_recovery.py
```

前两条分别展示行走和起身；第三条使用这两个保留策略执行行走／跌倒恢复切换。
