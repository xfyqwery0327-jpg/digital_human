@echo off
rem Switch CosyVoice 2 or 3. Example: switch-cosyvoice.bat 3
rem Restart narrator only: switch-cosyvoice.bat 3 -Restart
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0switch-cosyvoice.ps1" %*
