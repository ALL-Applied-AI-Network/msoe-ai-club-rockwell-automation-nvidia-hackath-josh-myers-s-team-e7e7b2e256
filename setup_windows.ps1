$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir
$env:UV_CACHE_DIR = if ($env:UV_CACHE_DIR) { $env:UV_CACHE_DIR } else { Join-Path $RootDir ".uv-cache" }

Write-Host "== Physical AI Innovation Lab setup (Windows) =="

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

Write-Host "Python development setup complete. Build the Singularity image with docker/setup_singularity.sh on ROSIE when needed."
