@echo off
REM Merges every per-screenshot report into src\output\combined_report.html.
REM Pass --sort weakest to surface the least-confident screenshots first.
cd /d "%~dp0"
py -m hue_map.combine %*
