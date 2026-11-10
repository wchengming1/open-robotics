param(
    [ValidateSet('teacher', 'velstand')][string]$Stage = 'teacher',
    [ValidateRange(1, 4096)][int]$NumEnvs = 64,
    [ValidateRange(1, 50000)][int]$Iterations = 5,
    [string]$RunName = 'hd1910_recovery_smoke',
    [string]$ResumeRun = '',
    [string]$ResumeCheckpoint = '',
    [string]$StandTeacher = 'models/hd1910_stand_teacher.pt',
    [string]$WalkTeacher = 'logs/rsl_rl/microduck_hd1910_velocity/2026-10-20_19-52-31_hd1910_resume_20261020_195213/model_4999.pt',
    [switch]$ExportOnCompletion
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
$cacheRoot = Join-Path (Get-Location) 'logs/local_cache'
foreach ($folder in @('tmp', 'warp', 'cuda')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot $folder) | Out-Null
}
$env:TEMP = Join-Path $cacheRoot 'tmp'
$env:TMP = $env:TEMP
$env:WARP_CACHE_PATH = Join-Path $cacheRoot 'warp'
$env:CUDA_CACHE_PATH = Join-Path $cacheRoot 'cuda'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:WANDB_MODE = 'disabled'
$resumeRequested = ($ResumeRun -ne '' -and $ResumeCheckpoint -ne '')
if (($ResumeRun -ne '') -ne ($ResumeCheckpoint -ne '')) {
    throw 'Specify ResumeRun and ResumeCheckpoint together.'
}
if ($Stage -eq 'teacher') {
    $task = 'Mjlab-StandUp-Flat-MicroDuck-HD1910'
    $experiment = 'microduck_hd1910_stand_teacher'
    $resume = 'False'
    $env:MICRODUCK_WARM_START = '0'
} else {
    $task = 'Mjlab-VelStand-Flat-MicroDuck-HD1910'
    $experiment = 'microduck_hd1910_velstand'
    foreach ($required in @($StandTeacher, $WalkTeacher)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required local HD1910 teacher missing: $required"
        }
    }
    if ((Resolve-Path -LiteralPath $StandTeacher).Path -ne (Join-Path (Get-Location) 'models/hd1910_stand_teacher.pt')) {
        Copy-Item -LiteralPath $StandTeacher -Destination 'models/hd1910_stand_teacher.pt' -ErrorAction Stop
    }
    if ((Resolve-Path -LiteralPath $WalkTeacher).Path -ne (Join-Path (Get-Location) 'models/hd1910_walk_teacher.pt')) {
        Copy-Item -LiteralPath $WalkTeacher -Destination 'models/hd1910_walk_teacher.pt' -ErrorAction Stop
    }
    if (-not $resumeRequested) {
        $seedDir = "logs/rsl_rl/$experiment/seed_hd1910_walk"
        New-Item -ItemType Directory -Force -Path $seedDir | Out-Null
        Copy-Item -LiteralPath $WalkTeacher -Destination (Join-Path $seedDir 'model_walk.pt')
    }
    $resume = 'True'
    $env:MICRODUCK_WARM_START = '1'
}
if ($resumeRequested) {
    $resume = 'True'
    $env:MICRODUCK_WARM_START = '0'
    if (-not (Test-Path -LiteralPath "logs/rsl_rl/$experiment/$ResumeRun/$ResumeCheckpoint" -PathType Leaf)) {
        throw 'Resume checkpoint does not exist in this stage experiment.'
    }
}
Write-Host "VelStand pipeline stage=$Stage task=$task envs=$NumEnvs iterations=$Iterations"
$trainingArgs = @('-m', 'mjlab_microduck.train_cli', $task,
    '--env.scene.num-envs', "$NumEnvs", '--agent.max-iterations', "$Iterations",
    '--agent.resume', $resume, '--agent.logger', 'tensorboard', '--agent.run-name', $RunName)
if ($resumeRequested) {
    $trainingArgs += @('--agent.load-run', ('^'+[regex]::Escape($ResumeRun)+'$'),
        '--agent.load-checkpoint', ('^'+[regex]::Escape($ResumeCheckpoint)+'$'))
} elseif ($Stage -eq 'velstand') {
    $trainingArgs += @('--agent.load-run', '^seed_hd1910_walk$', '--agent.load-checkpoint', '^model_walk\.pt$')
}
& $python @trainingArgs
if ($LASTEXITCODE -ne 0) { throw "Recovery training failed (exit $LASTEXITCODE)." }
if ($ExportOnCompletion) {
    $directory = Get-ChildItem -LiteralPath "logs/rsl_rl/$experiment" -Directory |
        Where-Object { $_.Name.EndsWith("_$RunName") } | Sort-Object Name | Select-Object -Last 1
    if (-not $directory) { throw 'Training output directory missing.' }
    $checkpoint = Get-ChildItem -LiteralPath $directory.FullName -Filter 'model_*.pt' |
        Sort-Object { [int]($_.BaseName -replace '^model_', '') } | Select-Object -Last 1
    if (-not $checkpoint) { throw 'Training checkpoint missing.' }
    $onnx = Join-Path $directory.FullName 'policy.onnx'
    # Warm-start is for the training load only, never for checkpoint export.
    $env:MICRODUCK_WARM_START = '0'
    & $python scripts/export.py $task --checkpoint-file $checkpoint.FullName --onnx-file $onnx --num-envs 1 --device cuda:0
    if ($LASTEXITCODE -ne 0) { throw 'Recovery ONNX export failed.' }
    $verify = @'
import sys,numpy as np,onnxruntime as ort
s=ort.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider'])
assert s.get_inputs()[0].shape==[1,61]
y=s.run(None,{s.get_inputs()[0].name:np.zeros((1,61),dtype=np.float32)})[0]
assert y.shape==(1,14) and np.isfinite(y).all()
print('ONNX gate passed: 61 inputs, 14 finite outputs.')
'@
    & $python -c $verify $onnx
    if ($LASTEXITCODE -ne 0) { throw 'Recovery ONNX verification failed.' }
    Write-Host "Exported: $onnx"
}
