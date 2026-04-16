@echo off
REM Hue-Map pipeline launcher. Forwards everything to the Python CLI.
REM
REM Input options:
REM   run.bat                                  ->  processes every image in src\input\
REM   run.bat path\to\image.png                ->  processes a single image
REM   run.bat --base64 "iVBORw0KGgo..."        ->  decodes and processes an inline
REM                                                base64 string (accepts
REM                                                "data:image/png;base64,..." URIs)
REM   run.bat --base64-file payload.txt        ->  reads base64 from a file (use
REM                                                this when the string is too
REM                                                long for a single command)
REM   type image.png ^| run.bat --stdin        ->  pipes raw image bytes via stdin
REM
REM Extras:
REM   --label MYNAME       output folder name for base64/stdin inputs
REM   --references DIR     override reference library location
REM   --output DIR         override output directory
REM   --log-level DEBUG    verbose logging
cd /d "%~dp0"
py -m hue_map.pipeline %*
