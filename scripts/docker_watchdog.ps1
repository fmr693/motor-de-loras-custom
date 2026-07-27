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

PAUSA (importante): el watchdog NO distingue "Docker se ha caido" de "he cerrado
Docker a proposito" -> sin esto, cerrarlo a mano era inutil porque lo relanzaba
en <5 min. Para que se este quieto, crea el fichero marcador:

    scripts\watchdog_pausa.bat        (o crea a mano  logs\watchdog.pausa )

y para reactivarlo:

    scripts\watchdog_reanudar.bat     (o borra ese fichero)

Se eligio un marcador explicito en vez de que el script adivine la intencion:
predecible sobre magico. El estado se consulta en logs\docker_watchdog.log.

Uso manual:
    powershell -ExecutionPolicy Bypass -File scripts\docker_watchdog.ps1

Como tarea programada (cada 5 min, SIN ventana):
    OJO: -WindowStyle Hidden NO basta. powershell.exe crea la consola y luego la
    oculta, asi que con LogonType=Interactive se ve un destello cada 5 min. La
    tarea debe registrarse con LogonType=S4U (corre en segundo plano, sin sesion
    grafica) -> ninguna ventana, nunca. Ver scripts\README_scripts.md.

    (borrar con:  schtasks /Delete /TN "MotorDockerWatchdog" /F )

Idempotente: si Docker ya responde, no hace nada. Registra cada accion en
logs/docker_watchdog.log (junto al repo).
#>

$ErrorActionPreference = "SilentlyContinue"
$RepoRoot   = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogFile    = Join-Path $RepoRoot "logs\docker_watchdog.log"
$PausaFile  = Join-Path $RepoRoot "logs\watchdog.pausa"
$DesktopExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

function Write-Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

# 0. ¿Pausado a proposito? El watchdog no puede distinguir una caida de un
#    cierre deliberado, asi que el usuario lo declara con un marcador. Sin esto,
#    cerrar Docker a mano era imposible: se relanzaba solo en <5 min.
if (Test-Path $PausaFile) {
    # Se registra UNA vez por pausa (no cada 5 min) para no inundar el log.
    $ultima = Get-Content $LogFile -Tail 1 -ErrorAction SilentlyContinue
    if ($ultima -notmatch "PAUSADO") {
        Write-Log "PAUSADO por marcador (logs\watchdog.pausa). No se tocara Docker hasta borrarlo."
    }
    exit 0
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
