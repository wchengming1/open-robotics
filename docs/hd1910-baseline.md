# HD1910 双足训练基线

2026-11-10 恢复为原始 Microduck 双足结构、HD1910 舵机适配，保留行走和起身策略。
轮子及后续专项实验已移入备份，详见 [恢复记录](HD1910_RESTORED.md)。

## 舵机适配

- `actuator/feetech_bam.py`：初始化 BAM 飞特控制器的目标历史，并支持单环境重置。
- `tasks/microduck_hd1910_env_cfg.py`：复制原环境机器人配置，仅替换 14 个执行器。
- `tasks/microduck_hd1910_recovery_env_cfg.py`：原起身及 VelStand 环境的 HD1910 变体。
- `robot/hd1910/1910_m6.json`：LuwuDynamics 公开的 M6 参数，原样保留。

参数固定提交、SHA-256 和许可证见
[参数来源](../src/mjlab_microduck/robot/hd1910/README.md)。本项目未重新辨识舵机参数，
没有改变原 Microduck 几何、质量、惯量或 HOME 姿态。

训练使用 P=5，电压 7.4–8.0 V，压降下限 7.0 V，延迟 3–6 个 5 ms 仿真步。
控制为 50 Hz，actor 观测 61 维，动作 14 维。导出包含观测归一化。
HD1910 策略应使用配套回放入口，避免误用 `infer_policy.py` 默认的 XL330 动力学。

## 训练与回放

```powershell
# 先执行 64 环境、5 次迭代的冒烟测试。
powershell -ExecutionPolicy Bypass -File scripts/train_velocity.ps1 -Motor hd1910 -NumEnvs 64 -Iterations 5 -RunName hd1910_smoke -ExportOnCompletion
powershell -ExecutionPolicy Bypass -File scripts/train_hd1910_recovery.ps1 -Stage teacher -NumEnvs 64 -Iterations 5 -RunName hd1910_teacher_smoke -ExportOnCompletion

# 已训练策略的行走、起身及切换回放。
powershell -ExecutionPolicy Bypass -File scripts/view_hd1910.ps1
.venv/Scripts/python.exe scripts/view_standup_push.py
.venv/Scripts/python.exe scripts/view_walk_recovery.py
```

原 XL330 任务和上游自带的被动轮滑任务仍属原始源码。本地 2804 主动轮实验已移出。

## 回归检查

Windows 下启用 UTF-8，并将虚拟环境 Scripts 加入 PATH：

```powershell
$env:PYTHONUTF8='1'
$env:PATH=(Join-Path $PWD '.venv/Scripts')+';'+$env:PATH
.venv/Scripts/python.exe -m pytest tests/test_hd1910_cfg.py tests/test_hd1910_recovery_cfg.py tests/test_hd1910_rollout.py tests/test_hd1910_recovery_rollout.py -q
```

20 项 HD1910 检查通过，包含 GPU 步进、单环境重置和站姿高度／倾角。
策略仿真比较见 [models/SOURCE.md](../models/SOURCE.md)。这些结果不替代实机校准。
