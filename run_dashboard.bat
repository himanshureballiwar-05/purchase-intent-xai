@echo off
cd /d "%~dp0"
echo Starting dashboard... keep this window OPEN. Browser: http://localhost:8501
python -m streamlit run dashboard/app.py
pause
