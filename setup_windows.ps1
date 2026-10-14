$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $RootDir "scripts/setup_windows.ps1") @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
