@echo off
rem Install brAIn dump for the current user; see install.ps1.
cd /d "%TEMP%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
pause
