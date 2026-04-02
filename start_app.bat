@echo off
setlocal

REM Ensure script runs from project root
cd /d "%~dp0"

echo [1/4] Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo .venv not found, creating with Python 3.11...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3.11 -m venv .venv 2>nul
        if errorlevel 1 py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 goto :error
)

set "PYTHON_EXE=.venv\Scripts\python.exe"

echo [2/4] Installing dependencies...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto :error
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 goto :error
REM Safety net: make sure uvicorn exists in this venv
"%PYTHON_EXE%" -m pip install "uvicorn[standard]"
if errorlevel 1 goto :error
REM Optional acceleration: install CUDA torch if supported (Windows/Linux NVIDIA only)
if exist "tools\ensure_torch_accel.py" (
    "%PYTHON_EXE%" "tools\ensure_torch_accel.py"
)

echo [3/4] Starting backend (uvicorn)...
if not exist "run_backend.bat" (
    echo run_backend.bat not found.
    goto :error
)
start "Philosophy RAG Backend" cmd /k "cd /d ""%~dp0"" && call run_backend.bat"

echo [4/4] Opening frontend...
timeout /t 3 /nobreak >nul
if exist "frontend.html" (
    start "" "%~dp0frontend.html"
) else (
    start "" "http://127.0.0.1:8000"
)

echo.
echo Startup complete.
echo - Backend: http://127.0.0.1:8000
echo - Frontend: frontend.html
exit /b 0

:error
echo.
echo Startup failed. Please check the error log above.
pause
exit /b 1
