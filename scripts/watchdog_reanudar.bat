@echo off
REM Reactiva el MotorDockerWatchdog: vuelve a vigilar que Docker este arriba
REM y lo relanza si se cae. Deshace watchdog_pausa.bat.

set "MARCA=%~dp0..\logs\watchdog.pausa"

if exist "%MARCA%" (
    del "%MARCA%"
    echo.
    echo   [OK] Watchdog REACTIVADO.
    echo.
    echo   Volvera a vigilar Docker en la proxima pasada ^(max. 5 min^).
) else (
    echo.
    echo   El watchdog ya estaba activo ^(no habia marcador de pausa^).
)
echo.
pause
