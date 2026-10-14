$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir
$env:UV_CACHE_DIR = if ($env:UV_CACHE_DIR) { $env:UV_CACHE_DIR } else { Join-Path $RootDir ".uv-cache" }

Write-Host "== Physical AI Innovation Lab host setup (Windows) =="
Write-Host "Note: ROSIE Slurm, Isaac Sim standalone, and the Cosmos CUDA environment are Linux-only."
Write-Host "This script prepares the shared Python development environment; run setup_linux.sh on ROSIE for Problems 1-3."

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found; installing it for the current user..."
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv installation completed but uv is not on PATH. Add %USERPROFILE%\.local\bin to PATH and rerun this script."
    exit 1
}

New-Item -ItemType Directory -Force -Path "data/problem1", "data/problem2", "data/problem3", "configs", "artifacts", "models", "outputs", "logs" | Out-Null
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

uv sync --dev
uv run python main.py

Write-Host "Windows host setup complete. Run setup_linux.sh on ROSIE to install Isaac Sim and Cosmos for Problems 1-3."
