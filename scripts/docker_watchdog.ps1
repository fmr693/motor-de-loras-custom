<#
docker_watchdog.ps1
===================
Vigila que el engine de Docker responda y, si no, relanza Docker Desktop.

Contexto (ver PRUEBAS_ESTRES.md): Docker Desktop se cayo DOS veces en una sola
sesion de pruebas de estres. Es el eslabon fragil del stack soberano en Windows:
si el `npipe` deja de responder, todo el ecosistema (serve, Odysseus, Hermes)
queda inaccesible. Los contenedores tienen restart: unless-stopped, asi que en
cuanto el engine vuelve, ellos vuelven solos; este script solo garantiza que el
engine este arriba.

Uso manual:
    powershell -ExecutionPolicy Bypass -File scripts\docker_watchdog.ps1

Como tarea programada (cada 5 min):
    schtasks /Create /SC MINUTE /MO 5 /TN "MotorDockerWatchdog" ^
      /TR "powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File \"C:\Users\Felipe\Desktop\Proyecto\motor-de-loras-custom\scripts\docker_watchdog.ps1\""
    (borrar con:  schtasks /Delete /TN "MotorDockerWatchdog" /F )

Idempotente: si Docker ya responde, no hace nada. Registra cada accion en
logs/docker_watchdog.log (junto al repo).
#>

$ErrorActionPreference = "SilentlyContinue"
$RepoRoot   = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogFile    = Join-Path $RepoRoot "logs\docker_watchdog.log"
$DesktopExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

function Write-Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

# 1. ¿Responde el engine?
& docker version --format '{{.Server.Version}}' 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    # engine sano; nada que hacer (no ensuciamos el log en el caso normal)
    exit 0
}

Write-Log "Engine Docker NO responde."

# 2. ¿Esta corriendo el proceso de Docker Desktop?
$proc = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
if (-not $proc) {
    if (Test-Path $DesktopExe) {
        Write-Log "Docker Desktop no esta en ejecucion -> relanzando."
        Start-Process $DesktopExe
    } else {
        Write-Log "ERROR: no se encuentra Docker Desktop en $DesktopExe"
        exit 1
    }
} else {
    Write-Log "Docker Desktop esta en ejecucion pero el engine no responde (arrancando o colgado)."
}

# 3. Esperar a que el engine vuelva (hasta ~3 min)
for ($i = 0; $i -lt 36; $i++) {
    Start-Sleep -Seconds 5
    & docker version --format '{{.Server.Version}}' 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Log ("Engine recuperado tras ~{0}s. Los contenedores (restart: unless-stopped) vuelven solos." -f ($i * 5 + 5))
        exit 0
    }
}

Write-Log "TIMEOUT: el engine no volvio en ~3 min. Revisar Docker Desktop a mano."
exit 1
