@echo off
rem  Double-click to build a standalone OmnisVox.exe (no Python needed to run the result).
rem  Output: dist\OmnisVox\OmnisVox.exe  and  OmnisVox-windows.zip  (copy either anywhere).
setlocal
cd /d "%~dp0"
call :setup || exit /b 1
echo.
echo  Building OmnisVox.exe (5-10 minutes)...
".venv\Scripts\python.exe" -m pip install "pyinstaller>=6.6" || goto build_failed
if exist build rmdir /s /q build
if exist dist\OmnisVox rmdir /s /q dist\OmnisVox
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --name OmnisVox ^
    --paths . --collect-submodules omnisvox_cfd --collect-submodules omnisvox_desktop ^
    --exclude-module pyqtgraph.examples --exclude-module pyqtgraph.opengl ^
    --exclude-module cupy --exclude-module tkinter ^
    packaging\launcher.py || goto build_failed
echo  Checking the built program...
"dist\OmnisVox\OmnisVox.exe" --selftest > build\selftest.txt 2>&1
findstr /c:"selftest: OK" build\selftest.txt >nul || (
    type build\selftest.txt
    goto build_failed
)
powershell -NoProfile -Command "Compress-Archive -Force -Path 'dist\OmnisVox' -DestinationPath 'OmnisVox-windows.zip'"
echo.
echo  Done.  Run:  dist\OmnisVox\OmnisVox.exe
echo  Portable copy:  OmnisVox-windows.zip  (unzip anywhere, run OmnisVox.exe inside)
echo  Note: the .exe is CPU-only. For the GPU, use OmnisVox.bat after "pip install cupy-cuda12x".
explorer "dist\OmnisVox"
pause
exit /b 0

:build_failed
echo.
echo  Build failed (see the messages above).
pause
exit /b 1

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
