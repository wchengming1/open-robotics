# HD1910M VelStand training

The existing HD1910 walking policy is retained. The integrated VelStand recipe
requires a recovery expert, so this workspace uses a local two-stage pipeline.
No author W&B checkpoint or XL330 teacher is used.

1. Train `Mjlab-StandUp-Flat-MicroDuck-HD1910` from random initialization, using
   the original ground-state curriculum (standing/sitting first, then prone and
   supine). Initial budget: 6000 iterations; choose further training from measured
   recovery results, not total reward alone.
2. Evaluate per-spawn recovery and standing body control. A completed training
   budget alone does not qualify the checkpoint as a teacher.
3. The restored selected teacher is `models/hd1910_stand_teacher.pt` (iteration
   5999); the walking checkpoint is `models/hd1910_walk_teacher.pt` (iteration
   4999). There is no completed local integrated VelStand policy in this restored
   baseline. `scripts/view_walk_recovery.py` switches between the two separate
   ONNX policies. For future integrated training, warm-start
   `Mjlab-VelStand-Flat-MicroDuck-HD1910` from the HD1910 walking checkpoint.
   The algorithm uses the teacher's own normalizer for fallen frames and the
   frozen HD1910 walk anchor for upright frames.

Both task families preserve the base robot/collision model, 61D actor interface,
14 servo actions, 50 Hz control, BAM friction DR and sensor conventions. Only the
actuators are replaced with the established HD1910 M6 baseline (P=5, 7.4-8.0 V).
Existing XL330 and VelStand factories are not modified.

## Launching

Run the 64-environment smoke test before long runs:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/train_hd1910_recovery.ps1 -Stage teacher -NumEnvs 64 -Iterations 5 -RunName hd1910_teacher_smoke -ExportOnCompletion
```

The recovery launcher defaults to the teacher stage. It does not automatically
promote the last checkpoint to an expert or launch integration without evaluation.
Integration uses `-Stage velstand -StandTeacher <validated checkpoint>`. The walk
checkpoint defaults to the final local `model_4999.pt`. Warm-start resets the new
student's curricula while retaining its weights, normalizers and optimizer; the
inherited walking curricula are fixed at their final stage by the base recipe.

Teacher logs: `logs/rsl_rl/microduck_hd1910_stand_teacher/`.
Integrated student logs: `logs/rsl_rl/microduck_hd1910_velstand/`.
