@echo off
setlocal
cd /d "%~dp0"
echo ==========================================
echo   VYAKARANA-X - first-time setup / launch
echo ==========================================
where py >nul 2>nul
if errorlevel 1 (echo Python launcher not found. Install Python 3.12 and try again.& pause& exit /b 1)
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 (echo Could not create Python 3.12 environment.& pause& exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (echo Python dependencies failed to install.& pause& exit /b 1)
if not exist "data\processed\mlm_train.txt" (
  echo Downloading open datasets. This requires internet and may take a few minutes...
  python scripts\download_data.py --max-lines 100000
  python scripts\prepare_data.py
)
if not exist "frontend\node_modules" (
  where npm >nul 2>nul
  if errorlevel 1 (echo Node.js 20+ / npm not found. Install Node.js and relaunch.& pause& exit /b 1)
  cd frontend
  call npm install
  cd ..
)
start "VYAKARANA-X API" cmd /k "cd /d ""%~dp0"" && call .venv\Scripts\activate.bat && uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000"
timeout /t 3 >nul
start "VYAKARANA-X FRONTEND" cmd /k "cd /d ""%~dp0frontend"" && npm run dev"
timeout /t 4 >nul
start http://localhost:5173
echo App launch commands sent. Keep both terminal windows open.
pause
