@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python train.py --epochs 3 --max-lines 100000
pause
