@echo off
rem Double-click to start the S.E.D.O.S.S. farmer hub.
rem First run: creates the project's Python environment and installs the packages - a few minutes.
rem Later runs: starts the hub and opens it in the browser. Close this window to stop the hub.

cd /d "%~dp0"
title S.E.D.O.S.S. farmer hub
echo S.E.D.O.S.S. farmer hub
echo.

if exist ".venv\Scripts\python.exe" goto check

set "PY="
for %%v in (3.14 3.13 3.12 3.11 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3.10 to 3.14 was not found.
  echo Install Python 3.14 from python.org/downloads - tick "Add python.exe to PATH" -
  echo then double-click this file again.
  goto fail
)

echo First run: setting up with %PY%. This takes a few minutes, please wait...
%PY% -m venv .venv
if errorlevel 1 (
  echo Could not create the Python environment.
  goto fail
)

:check
".venv\Scripts\python.exe" -c "import numpy, pandas, xgboost, yaml, sklearn, matplotlib, requests" >nul 2>&1
if not errorlevel 1 goto run

echo Installing the packages...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r ml\requirements.txt
if errorlevel 1 (
  echo Installing the packages failed. Check the internet connection and try again.
  goto fail
)
".venv\Scripts\python.exe" -c "import xgboost" >nul 2>&1
if errorlevel 1 (
  echo XGBoost could not load. Install the Microsoft Visual C++ Redistributable x64
  echo from Microsoft's website, then double-click this file again.
  goto fail
)
echo Setup finished.
echo.

:run
".venv\Scripts\python.exe" ml\run_bench.py --serve --open %*
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
