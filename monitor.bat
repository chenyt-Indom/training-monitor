@echo off
chcp 65001 >nul
title YOLO Training Monitor
set PYTHONIOENCODING=utf-8

REM 想固定监控某个目录，把下面这行改成:
REM   set RUNS=D:/fire_runs
REM 留空则自动探测当前目录下的 runs\detect 等位置。
set RUNS=

if "%RUNS%"=="" (
  python monitor.py
) else (
  python monitor.py --runs-dir "%RUNS%"
)

echo.
echo 监控已退出。训练进程不受影响。
pause
