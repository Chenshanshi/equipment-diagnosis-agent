@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
echo 正在启动设备诊断智能体，浏览器会自动打开 http://127.0.0.1:8770 ...
echo 关闭这个窗口即停止服务。
start "" cmd /c "timeout /t 8 >nul & start http://127.0.0.1:8770/"
%PY% diag_agent\app.py 8770
pause
