<#
.SYNOPSIS
    PowerShell wrapper to batch preprocess videos using preprocess_videos.py

.DESCRIPTION
    Recursively finds common video files under an input folder and runs the
    Python preprocessing script in parallel worker processes. Designed for
    Windows users with ffmpeg and Python on PATH.

.PARAMETER InputDir
    Directory containing raw videos.

.PARAMETER OutputDir
    Directory where preprocessed videos will be written.

.PARAMETER Fps
    Target framerate (default 10)

.PARAMETER Width
    Target width in pixels (default 640)

.PARAMETER Height
    Target height in pixels (default 480)

.PARAMETER Workers
    Number of parallel worker processes to use for CPU-bound ffmpeg calls.

.EXAMPLE
    .\preprocess_videos.ps1 -InputDir .\data\raw_videos -OutputDir .\data\videos -Workers 4 -Denoise
#>

param(
    [Parameter(Mandatory=$true)] [string] $InputDir,
    [Parameter(Mandatory=$true)] [string] $OutputDir,
    [int] $Fps = 10,
    [int] $Width = 640,
    [int] $Height = 480,
    [int] $Workers = 1,
    [switch] $Denoise,
    [switch] $NoRecursive
)

function Ensure-PythonScript {
    param([string] $ScriptPath)
    if (-not (Test-Path $ScriptPath)) {
        Write-Error "Required script $ScriptPath not found. Run from repository root where preprocess_videos.py exists."
        exit 1
    }
}

$scriptPath = Join-Path (Get-Location) 'preprocess_videos.py'
Ensure-PythonScript -ScriptPath $scriptPath

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found on PATH. Install Python and add to PATH."
    exit 1
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg not found on PATH. Install ffmpeg and add to PATH."
    exit 1
}

$rec = -not $NoRecursive

$den = $false
if ($Denoise) { $den = $true }

$argsBase = @(
    '--input_dir', (Resolve-Path $InputDir).Path,
    '--output_dir', (Resolve-Path $OutputDir).Path,
    '--fps', $Fps.ToString(),
    '--width', $Width.ToString(),
    '--height', $Height.ToString()
)

if ($den) { $argsBase += '--denoise' }
if (-not $rec) { $argsBase += '--no_recursive' }
if ($Workers -gt 1) { $argsBase += '--workers'; $argsBase += $Workers.ToString() }

Write-Host "Starting preprocessing:" -ForegroundColor Cyan
Write-Host "  Input:  " (Resolve-Path $InputDir).Path
Write-Host "  Output: " (Resolve-Path $OutputDir).Path
Write-Host "  FPS:    $Fps  Size: ${Width}x${Height}  Workers: $Workers  Denoise: $den"

# Build final argument line
$argLine = $argsBase -join ' '

try {
    & python $scriptPath $argsBase
    if ($LASTEXITCODE -ne 0) { throw "preprocess_videos.py exited with code $LASTEXITCODE" }
}
catch {
    Write-Error "Preprocessing failed: $_"
    exit 1
}

Write-Host "Preprocessing finished." -ForegroundColor Green
