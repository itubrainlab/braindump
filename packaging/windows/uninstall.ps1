# Remove brAIn stream for the current user. Projects, recordings and the list
# of recent projects (%APPDATA%\brAIn stream) are kept. Started by
# uninstall.cmd.
$ErrorActionPreference = 'Stop'

$data = Join-Path $env:LOCALAPPDATA 'brAIn stream'
$link = Join-Path ([Environment]::GetFolderPath('Programs')) 'brAIn stream.lnk'
if (Test-Path $link) { Remove-Item -Force $link }

if (Test-Path $data) {
    if ($PSScriptRoot.StartsWith($data, [StringComparison]::OrdinalIgnoreCase)) {
        # Run from the installed copy: its uninstall.cmd is still open, so
        # remove the folder once that has finished.
        Start-Process -WindowStyle Hidden cmd.exe -ArgumentList `
            "/c timeout /t 3 /nobreak >nul & rmdir /s /q `"$data`""
    } else {
        Remove-Item -Recurse -Force $data
    }
}
Write-Host 'brAIn stream is removed. Projects and recordings are kept.'
