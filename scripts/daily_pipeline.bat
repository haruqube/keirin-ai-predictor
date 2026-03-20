@echo off
call C:\Users\consu\anaconda3\Scripts\activate.bat
cd /d C:\Users\consu\desktop\sato\claude\projects\keirin-ai-predictor
python scripts/daily_pipeline.py %*
