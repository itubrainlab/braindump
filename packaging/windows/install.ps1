# Install brAIn stream for the current user: copies this folder to
# %LOCALAPPDATA%\brAIn stream\app, adds a Start menu shortcut with the icon,
# then sets it up (downloads g.Pype and the libraries from PyPI). No
# administrator rights needed. Started by install.cmd.
$ErrorActionPreference = 'Stop'

if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'brAIn stream needs 64-bit Windows.'
}

$here = $PSScriptRoot
$data = Join-Path $env:LOCALAPPDATA 'brAIn stream'
$app = Join-Path $data 'app'

if ($here.TrimEnd('\') -ne $app) {
    Write-Host "Copying brAIn stream to $app"
    if (Test-Path $app) { Remove-Item -Recurse -Force $app }
    New-Item -ItemType Directory -Force -Path $app | Out-Null
    Copy-Item -Recurse -Force (Join-Path $here '*') $app
}
# Files unpacked from a downloaded zip carry its mark of the web, which
# makes Windows ask before running each program.
Get-ChildItem -Recurse -File $app | Unblock-File

$programs = [Environment]::GetFolderPath('Programs')
$link = (New-Object -ComObject WScript.Shell).CreateShortcut(
    (Join-Path $programs 'brAIn stream.lnk'))
$link.TargetPath = Join-Path $app 'python\pythonw.exe'
$link.Arguments = '-I "' + (Join-Path $app 'launcher.py') + '"'
$link.WorkingDirectory = $app
$link.IconLocation = Join-Path $app 'brainstream.ico'
$link.Description = 'Record EEG, eye tracking, ECG, GSR and LSL streams into BIDS'
$link.Save()

Write-Host 'Setting up (downloads g.Pype and the libraries; a few minutes)...'
$build = (Get-Content (Join-Path $app 'build-id') -Raw).Trim()
& (Join-Path $app 'python\python.exe') -I (Join-Path $app 'bootstrap.py') `
    $app (Join-Path $data "runtime\$build")
if ($LASTEXITCODE -ne 0) {
    throw "Setup failed. See $(Join-Path $data 'runtime\setup.log')"
}

Write-Host ''
Write-Host 'brAIn stream is installed: open it from the Start menu.'
