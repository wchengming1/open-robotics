param([string]$Onnx = 'models/hd1910_velocity_final.onnx')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:WARP_CACHE_PATH = Join-Path (Get-Location) 'logs/local_cache/warp'
$env:CUDA_CACHE_PATH = Join-Path (Get-Location) 'logs/local_cache/cuda'
$env:TEMP = Join-Path (Get-Location) 'logs/local_cache/tmp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
Write-Host 'Click the MuJoCo VIEWER window to control the robot:'
Write-Host '1 stand | 2 forward | 3 backward | 4 left | 5 right | 6 turn left | 7 turn right'
& .venv/Scripts/python.exe scripts/eval_hd1910.py --onnx $Onnx --view
if ($LASTEXITCODE -ne 0) { throw "HD1910 viewer exited with $LASTEXITCODE" }
