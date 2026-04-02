@echo off
echo Starting Plant Disease Detection with Tamil NLP...
echo.
echo Tamil translation will be enabled.
echo First Tamil translation will download the model (1-2 GB).
echo.
set ENABLE_TAMIL_NLP=true
set NLP_LIGHTWEIGHT=false
python app.py
