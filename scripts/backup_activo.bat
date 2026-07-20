@echo off
REM Wrapper del backup del activo para la tarea programada (evita el infierno de
REM comillas de schtasks). Edita --dest para apuntar a otro disco o a la nube.
REM Uso manual:  scripts\backup_activo.bat
REM Nota: Task Scheduler corre con un PATH minimo -> ruta COMPLETA de python.
REM       Si reinstalas python en otra ruta, actualiza PY aqui.
set REPO=C:\Users\Felipe\Desktop\Proyecto\motor-de-loras-custom
set MEM=C:\Users\Felipe\.claude\projects\c--Users-Felipe-Desktop-Proyecto-motor-de-loras-custom\memory
set PY=C:\Python313\python.exe
if not exist "%PY%" set PY=python
"%PY%" "%REPO%\scripts\backup_activo.py" --memory-dir "%MEM%"
