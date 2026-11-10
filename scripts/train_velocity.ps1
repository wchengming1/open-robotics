param(
    [ValidateSet('xl330', 'hd1910')][string]$Motor = 'hd1910',
    [ValidateRange(1, 4096)][int]$NumEnvs = 64,
    [ValidateRange(1, 50000)][int]$Iterations = 5,
    [string]$RunName = 'velocity_scratch_smoke',
    [string]$ResumeRun = '',
    [string]$ResumeCheckpoint = '',
    [switch]$ExportOnCompletion
)

# Independent of the VelStand launcher and all author checkpoints.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run uv sync first.' }
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
$task = 'Mjlab-Velocity-Flat-MicroDuck'
$experiment = 'velocity'
if ($Motor -eq 'hd1910') {
    $task += '-HD1910'
    $experiment = 'microduck_hd1910_velocity'
}

$resume = ($ResumeRun -ne '' -and $ResumeCheckpoint -ne '')
if (($ResumeRun -ne '') -ne ($ResumeCheckpoint -ne '')) {
    throw 'Specify ResumeRun and ResumeCheckpoint together.'
}
$trainingArgs = @(
    '-m', 'mjlab_microduck.train_cli', $task,
    '--env.scene.num-envs', "$NumEnvs",
    '--agent.max-iterations', "$Iterations",
    '--agent.resume', "$resume",
    '--agent.logger', 'tensorboard',
    '--agent.run-name', $RunName
)
if ($resume) {
    if ($env:MICRODUCK_WARM_START -eq '1') {
        throw 'Unset MICRODUCK_WARM_START to preserve the checkpoint curricula.'
    }
    $resumePath = Join-Path "logs/rsl_rl/$experiment/$ResumeRun" $ResumeCheckpoint
    if (-not (Test-Path -LiteralPath $resumePath -PathType Leaf)) {
        throw "Checkpoint not found: $resumePath"
    }
    $trainingArgs += @(
        '--agent.load-run', ('^' + [regex]::Escape($ResumeRun) + '$'),
        '--agent.load-checkpoint', ('^' + [regex]::Escape($ResumeCheckpoint) + '$')
    )
}
Write-Host "Motor=$Motor Task=$task Environments=$NumEnvs Iterations=$Iterations Resume=$resume Logger=tensorboard"
& $python @trainingArgs
if ($LASTEXITCODE -ne 0) { throw "Training failed (exit $LASTEXITCODE)." }

if ($ExportOnCompletion) {
    $directory = Get-ChildItem -LiteralPath "logs/rsl_rl/$experiment" -Directory |
        Where-Object { $_.Name.EndsWith("_$RunName") } |
        Sort-Object Name | Select-Object -Last 1
    if (-not $directory) { throw 'Training output directory was not found.' }
    $checkpoint = Get-ChildItem -LiteralPath $directory.FullName -Filter 'model_*.pt' |
        Sort-Object { [int]($_.BaseName -replace '^model_', '') } |
        Select-Object -Last 1
    if (-not $checkpoint) { throw 'No checkpoint was saved.' }
    $onnx = Join-Path $directory.FullName 'policy.onnx'
    & $python scripts/export.py $task --checkpoint-file $checkpoint.FullName `
        --onnx-file $onnx --num-envs 1 --device cuda:0
    if ($LASTEXITCODE -ne 0) { throw "Export failed (exit $LASTEXITCODE)." }
    $verify = @'
import sys
import numpy as np
import onnxruntime as ort
s = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
assert s.get_inputs()[0].shape == [1, 61], s.get_inputs()[0].shape
y = s.run(None, {s.get_inputs()[0].name: np.zeros((1, 61), dtype=np.float32)})[0]
assert y.shape == (1, 14) and np.isfinite(y).all(), y
print('ONNX gate passed: 61 inputs, 14 finite outputs. Rollout evaluation still required.')
'@
    & $python -c $verify $onnx
    if ($LASTEXITCODE -ne 0) { throw 'ONNX verification failed.' }
    Write-Host "Exported: $onnx"
}
