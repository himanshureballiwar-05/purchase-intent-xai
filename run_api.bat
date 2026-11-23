@echo off
cd /d "%~dp0"
echo Starting API... keep this window OPEN. Docs: http://localhost:8000/docs
python -m uvicorn src.api.main:app
pause
