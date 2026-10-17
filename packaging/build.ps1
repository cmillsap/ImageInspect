<#
.SYNOPSIS
    Builds the Image Inspect installers.

.DESCRIPTION
    For each variant: makes a clean build venv (build\venv-<variant>) with
    that variant's requirements and PyInstaller, freezes the app into
    dist\<variant>\ImageInspect, then has Inno Setup pack it into
    dist\ImageInspect-<version>[-GPU]-Setup.exe.

    The build venvs are kept between runs and only topped up, and pip's cache
    holds the large wheels, so later builds are quicker. The project's own
    venv is left alone.

.PARAMETER Variant
    cpu, cuda, or all (the default).

.PARAMETER SkipApp
    Reuse the existing dist\<variant>\ImageInspect and only recompile the
    installer.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\build.ps1
    powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Variant cpu
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda', 'all')]
    [string]$Variant = 'all',
    [switch]$SkipApp
)

$ErrorActionPreference = 'Stop'

$PackagingDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot  = Split-Path -Parent $PackagingDir
$BuildDir     = Join-Path $ProjectRoot 'build'
$DistDir      = Join-Path $ProjectRoot 'dist'
$Requirements = @{
    cpu  = Join-Path $ProjectRoot 'requirements.txt'
    cuda = Join-Path $ProjectRoot 'requirements-cuda.txt'
}

function Write-Stage($Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked($Description) {
    if ($LASTEXITCODE -ne 0) { throw "$Description failed." }
}

$Version = (Select-String -Path (Join-Path $ProjectRoot 'version.py') `
    -Pattern '__version__\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
Write-Host "Image Inspect $Version"

$IsccCandidates = @(
    (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $Iscc) {
    throw "Inno Setup 6 not found. Install it with: winget install JRSoftware.InnoSetup"
}

$Variants = if ($Variant -eq 'all') { @('cpu', 'cuda') } else { @($Variant) }
foreach ($V in $Variants) {
    $AppDir = Join-Path $DistDir "$V\ImageInspect"

    if (-not $SkipApp) {
        $Venv = Join-Path $BuildDir "venv-$V"
        $Python = Join-Path $Venv 'Scripts\python.exe'
        if (-not (Test-Path $Python)) {
            Write-Stage "Creating build venv ($V)"
            & py -3.12 -m venv $Venv
            Invoke-Checked 'Creating the venv'
        }
        Write-Stage "Installing packages ($V)"
        & $Python -m pip install --disable-pip-version-check -q `
            -r $Requirements[$V] -r (Join-Path $PackagingDir 'requirements-build.txt')
        Invoke-Checked 'pip install'

        Write-Stage "Running tests ($V)"
        & $Python -m pytest -q -p no:cacheprovider $ProjectRoot
        Invoke-Checked 'The tests'

        Write-Stage "Freezing the app ($V)"
        & $Python -m PyInstaller (Join-Path $PackagingDir 'ImageInspect.spec') `
            --noconfirm --log-level WARN `
            --distpath (Join-Path $DistDir $V) `
            --workpath (Join-Path $BuildDir "pyinstaller-$V")
        Invoke-Checked 'PyInstaller'
    }

    if (-not (Test-Path (Join-Path $AppDir 'ImageInspect.exe'))) {
        throw "Expected $AppDir\ImageInspect.exe. Run without -SkipApp to build it."
    }

    Write-Stage "Compiling the installer ($V)"
    & $Iscc /Q "/DAppVersion=$Version" "/DVariant=$V" (Join-Path $PackagingDir 'installer.iss')
    Invoke-Checked 'Inno Setup'
}

Write-Stage 'Done'
Get-ChildItem (Join-Path $DistDir "ImageInspect-$Version-*Setup.exe") | ForEach-Object {
    Write-Host ("    {0}  ({1:N0} MB)" -f $_.FullName, ($_.Length / 1MB))
}
