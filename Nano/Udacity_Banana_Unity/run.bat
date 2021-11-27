@echo off
for %%i in (1 2 3 4 5 6 7 8 ) ;do (
    python main.py --mode compare --type %%i )