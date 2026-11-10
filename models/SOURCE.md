# Retained HD1910 biped policies

Restored on 2026-11-10 to the upstream Microduck biped source plus the HD1910
servo integration, retaining walking and stand-up training results.

| Role | ONNX with baked normalizer | Matching checkpoint | Source iteration |
| --- | --- | --- | --- |
| Walking | `hd1910_velocity_final.onnx` | `hd1910_walk_teacher.pt` | 4999 |
| Stand-up | `hd1910_standup_best.onnx` | `hd1910_stand_teacher.pt` | 5999 |

Walking source run:
`logs/rsl_rl/microduck_hd1910_velocity/2026-10-20_19-52-31_hd1910_resume_20261020_195213/`.
Stand-up source run:
`logs/rsl_rl/microduck_hd1910_stand_teacher/2026-10-25_13-07-42_hd1910_teacher_resume_20261025_130719/`.

Both policies use the original biped geometry, 14 HD1910 servos, 61 observation
values and 14 joint-position actions at 50 Hz. There are no active wheel joints
or wheel PID/setpoint channels. Use the matching HD1910 BAM model, not the
upstream XL330 runtime defaults. Servo parameter attribution remains in
`src/mjlab_microduck/robot/hd1910/README.md`.

The walking ONNX is the original final training export. The stand-up ONNX was
exported from the retained checkpoint through `scripts/export.py`. Numerical
checks against checkpoint weights AND normalization passed for 32 random inputs
per policy; maximum absolute differences were below 8e-6.

## Selection measurements

Nominal GPU simulation, fixed seed 42, no pushes or observation noise; BAM
voltage randomization and actuator delays retained. This is finite simulation
validation, not hardware validation or proof of global optimality.

- Walking 4750 and 4999: each survived 56/56 ten-second trials across idle,
  forward/backward, lateral and turning commands. The normalized tracking MSE
  was 0.6411 and 0.6093 respectively, so 4999 is retained. Surviving a lateral
  command does not establish a useful lateral gait; see per-case measurements.
- Stand-up 3000, 4750, 5500, 5750 and 5999 each passed 64/64 fifteen-second
  trials (16 each prone, supine with roll variation, seated, upright); 5250
  passed 44/64. The final 5999 checkpoint is retained among the tied candidates.
  Success requires both feet contacting, height > 0.095 m, tilt < 20 degrees,
  planar speed < 0.15 m/s and angular speed < 1 rad/s for two continuous seconds.

Checksums, exact checkpoint paths and numeric export checks: `hd1910_policies.json`.
Candidate measurements: `logs/hd1910_baseline_validation/`.

## Replay

```powershell
powershell -ExecutionPolicy Bypass -File scripts/view_hd1910.ps1
.venv/Scripts/python.exe scripts/view_standup_push.py
.venv/Scripts/python.exe scripts/view_walk_recovery.py
```

Wheel policies, later experiments and unused XL330 downloads were moved to the
rollback backup; they are not selected by these replay commands.
