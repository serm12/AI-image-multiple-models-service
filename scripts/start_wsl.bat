@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

for /f "delims=" %%i in ('wsl wslpath -a "%PROJECT_DIR%"') do set "WSL_DIR=%%i"

if not defined WSL_DIR (
  echo [ERROR] Failed to resolve project path to WSL path.
  echo Please ensure WSL is installed and available.
  pause
  exit /b 1
)

echo Starting service in WSL...
wsl bash -lc "cd \"%WSL_DIR%\" && bash ./scripts/start_wsl.sh"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Service exited with code %EXIT_CODE%.
)

pause
exit /b %EXIT_CODE%
