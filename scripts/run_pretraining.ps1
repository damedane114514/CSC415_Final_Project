param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$configs = @(
    "configs/pretraining/cat_c4_pretrain.yaml",
    "configs/pretraining/cat_c8_pretrain.yaml",
    "configs/pretraining/cat_c16_pretrain.yaml"
)

foreach ($cfg in $configs) {
    Write-Host "\n=== Running pretraining config: $cfg ===" -ForegroundColor Cyan
    & $PythonExe "$ProjectRoot/scripts/pretrain.py" --config "$ProjectRoot/$cfg"
}

Write-Host "\nAll pretraining runs completed." -ForegroundColor Green
