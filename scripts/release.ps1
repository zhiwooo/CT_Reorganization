param(
    [string]$Version = "0.1.0",
    [switch]$IncludeSampleData,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectRootPath = $ProjectRoot.Path
$AppName = "TinyCT"
$ReleaseName = "$AppName-v$Version-windows-x64"
$ReleaseRoot = Join-Path $ProjectRootPath "release"
$StageDir = Join-Path $ReleaseRoot $ReleaseName
$ArchivePath = Join-Path $ReleaseRoot "$ReleaseName.zip"
$DistAppDir = Join-Path $ProjectRootPath "dist\$AppName"

function Assert-InProject {
    param([string]$Path)
    $Resolved = Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue
    if ($null -eq $Resolved) {
        return
    }
    if (-not $Resolved.Path.StartsWith($ProjectRootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify path outside project: $($Resolved.Path)"
    }
}

function Remove-ProjectPath {
    param([string]$Path)
    Assert-InProject $Path
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

Set-Location $ProjectRootPath

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $UvPath = Join-Path $env:USERPROFILE ".local\bin"
    $env:Path = "$UvPath;$env:Path"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install uv first, then rerun this script."
}

Write-Host "==> Syncing dependencies"
uv sync --dev

if (-not $SkipTests) {
    Write-Host "==> Running tests"
    uv run pytest -q
}

Write-Host "==> Cleaning previous build outputs"
Remove-ProjectPath (Join-Path $ProjectRootPath "build")
Remove-ProjectPath (Join-Path $ProjectRootPath "dist")
Remove-ProjectPath $StageDir
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

Write-Host "==> Building executable with PyInstaller"
uv run pyinstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name $AppName `
    --runtime-hook "scripts\pyi_runtime_hook.py" `
    --collect-all astra `
    --hidden-import PySide6.QtCore `
    --hidden-import PySide6.QtGui `
    --hidden-import PySide6.QtWidgets `
    --exclude-module pytest `
    "src\tiny_ct_app\main.py"

if (-not (Test-Path -LiteralPath (Join-Path $DistAppDir "$AppName.exe"))) {
    throw "PyInstaller did not produce $AppName.exe"
}

Write-Host "==> Smoke testing packaged executable"
$PackageExe = Join-Path $DistAppDir "$AppName.exe"
$PreviousQtPlatform = $env:QT_QPA_PLATFORM
$env:QT_QPA_PLATFORM = "offscreen"
$SmokeProcess = Start-Process `
    -FilePath $PackageExe `
    -ArgumentList "--smoke-test" `
    -WindowStyle Hidden `
    -PassThru
if (-not $SmokeProcess.WaitForExit(30000)) {
    Stop-Process -Id $SmokeProcess.Id -Force
    throw "Packaged executable smoke test timed out."
}
if ($SmokeProcess.ExitCode -ne 0) {
    throw "Packaged executable smoke test failed with exit code $($SmokeProcess.ExitCode)"
}
$env:QT_QPA_PLATFORM = $PreviousQtPlatform

Write-Host "==> Staging release files"
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
Copy-Item -LiteralPath $DistAppDir -Destination (Join-Path $StageDir $AppName) -Recurse
Copy-Item -LiteralPath (Join-Path $ProjectRootPath "README.md") -Destination (Join-Path $StageDir "README.md")
Copy-Item -LiteralPath (Join-Path $ProjectRootPath "RELEASE.md") -Destination (Join-Path $StageDir "RELEASE.md")
"$Version" | Set-Content -LiteralPath (Join-Path $StageDir "VERSION.txt") -Encoding UTF8

if ($IncludeSampleData) {
    $SampleSource = Join-Path $ProjectRootPath "proj"
    if (Test-Path -LiteralPath $SampleSource) {
        Write-Host "==> Including sample projection data"
        $SampleRoot = Join-Path $StageDir "sample_data"
        New-Item -ItemType Directory -Force -Path $SampleRoot | Out-Null
        Copy-Item -LiteralPath $SampleSource -Destination (Join-Path $SampleRoot "proj") -Recurse
    }
}

Write-Host "==> Creating archive"
Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $ArchivePath -Force

Write-Host ""
Write-Host "Release package created:"
Write-Host $ArchivePath
