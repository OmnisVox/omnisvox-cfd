@echo off
rem  Double-click to open OmnisVox CFD.  First run installs everything into .venv here.
setlocal
cd /d "%~dp0"
call :setup || exit /b 1
start "" ".venv\Scripts\pythonw.exe" -m omnisvox_desktop
exit /b 0

:setup
if exist ".venv\omnisvox_ready.txt" exit /b 0
echo.
echo  First-time setup: installing OmnisVox and its libraries.
echo  This needs internet and takes a few minutes. It only happens once.
echo.
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
if not defined PY (
    echo  Python 3.10 or newer was not found.
    echo  Install it from https://www.python.org/downloads/windows/
    echo  ^(tick "Add python.exe to PATH" in the installer^), then run this again.
    echo.
    pause
    exit /b 1
)
if exist ".venv" rmdir /s /q ".venv"
%PY% -m venv .venv || goto setup_failed
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto setup_failed
".venv\Scripts\python.exe" -m pip install -e ".[desktop]" || goto setup_failed
echo ok> ".venv\omnisvox_ready.txt"
echo  Setup finished.
exit /b 0

:setup_failed
echo.
echo  Setup failed (see the messages above). Check your internet connection and run this again.
pause
exit /b 1
