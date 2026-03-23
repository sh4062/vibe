$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

python (Join-Path $ScriptDir "prepare_dataset.py")
