@echo off
REM Processes every screenshot in src\input\ and writes reports into src\output\.
REM Pass a path to operate on a single file, e.g. run.bat ..\test\my_shot.png.
cd /d "%~dp0"
py -m hue_map.pipeline %*
