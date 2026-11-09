@echo off
rem Remove brAIn dump for the current user; see uninstall.ps1.
cd /d "%TEMP%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1"
pause
