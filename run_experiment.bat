@echo off
echo Starting RelBench MPNN Experiment...
set PYTHONPATH=src
.\.venv\Scripts\python.exe -u src\r2g_eval\main.py relbench
pause
