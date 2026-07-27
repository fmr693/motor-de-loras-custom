@echo off
REM Pausa el MotorDockerWatchdog: deja de relanzar Docker Desktop.
REM Usalo cuando cierres Docker A PROPOSITO (para liberar VRAM, para entrenar,
REM o simplemente porque no quieres el stack encendido).
REM Reactivar con: watchdog_reanudar.bat

set "MARCA=%~dp0..\logs\watchdog.pausa"
if not exist "%~dp0..\logs" mkdir "%~dp0..\logs"
echo Pausado el %DATE% a las %TIME% > "%MARCA%"

echo.
echo   [OK] Watchdog PAUSADO.
echo.
echo   Docker ya NO se relanzara solo. Puedes cerrarlo tranquilo.
echo   Marcador: %MARCA%
echo.
echo   Para volver a activarlo:  watchdog_reanudar.bat
echo.
pause
