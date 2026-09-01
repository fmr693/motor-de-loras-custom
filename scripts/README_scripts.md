# scripts/ — utilidades de operación

Herramientas de consolidación del proyecto (punto 1 del roadmap: afianzar lo que hay).
Nacidas de las pruebas de estrés (ver `../PRUEBAS_ESTRES.md`).

---

## `backup_activo.py` — copia de seguridad del dato

El activo del proyecto es el **dataset**, y vive en carpetas gitignoradas sin copia.
Este script las respalda (log canónico, `datasets/`, `SESION.md`, opcionalmente la
memoria persistente y los adapters).

```bash
# backup a ../_backups_motor (junto al repo)
python scripts/backup_activo.py

# recomendado: incluir la memoria persistente y apuntar a la nube / otro disco
python scripts/backup_activo.py \
  --dest "C:/Users/<usuario>/OneDrive/backups_motor" \
  --memory-dir "C:/Users/<usuario>/.claude/projects/<slug-del-proyecto>/memory"
```

**Tarea programada semanal** (domingos 20:00):

```bat
schtasks /Create /SC WEEKLY /D SUN /ST 20:00 /TN "MotorBackupActivo" ^
  /TR "python \"C:\ruta\al\repo\scripts\backup_activo.py\" --memory-dir \"C:\Users\<usuario>\.claude\projects\<slug-del-proyecto>\memory\""
```

Borrar: `schtasks /Delete /TN "MotorBackupActivo" /F`

> **AVISO:** el destino por defecto está en el **mismo disco** que el proyecto — no
> protege contra un disco muerto. Para protección real, usa `--dest` en otro disco o
> una carpeta sincronizada a la nube (OneDrive/Drive).

---

## `chequeo_activo.py` — ¿cuánta señal de entrenamiento hay?

Roadmap punto 2: el uso diario deja señal en el `interaction_log`; el primer
entrenamiento de comportamiento se lanza al llegar al umbral (~1-2k SFT limpios o
~300-500 pares DPO). Este chequeo lo mide **en seco, sin GPU ni serve**.

```bash
python scripts/chequeo_activo.py
```

Reutiliza las mismas piezas que `learn` y `dpo` (`log_quality.load_sft_examples`,
`DPOBuilder.stats`), así que cuenta exactamente lo que entrenaría. Pensado para correr
semanalmente (o cuando apetezca ver el progreso). La reflexión FRESCA —que genera los
pares de corrección para DPO— sí necesita el serve (`fabrica_loras reflect`); el chequeo
solo lee lo que ya está en disco y avisa si falta.

> **Línea base (21 jul 2026):** 430 interacciones → **87 SFT limpios** (~6 % del umbral) y
> **0 pares DPO**. El 0 en DPO es estructural, no de volumen: hay 20 👍 y 17 👎 pero sobre
> prompts DISTINTOS, y un par de preferencia necesita el MISMO prompt con respuesta buena y
> mala. La vía real a DPO son los pares de corrección de la reflexión (usuario reformula →
> error), que hoy están a 0 en disco. **Conclusión: el uso solo no llena DPO; hay que correr
> `reflect` con el serve arriba, o —más adelante— un "regenerar al 👎" que dé el mismo prompt
> dos veces.**

---

## `docker_watchdog.ps1` — vigilante de Docker

Docker Desktop se cayó **dos veces** en una sola sesión de pruebas de estrés: es el
eslabón frágil del stack soberano en Windows. Este watchdog comprueba el engine y, si
no responde, relanza Docker Desktop. Los contenedores (`restart: unless-stopped`)
vuelven solos en cuanto el engine está arriba.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\docker_watchdog.ps1
```

### Pausarlo cuando cierres Docker a propósito

El watchdog **no distingue "Docker se ha caído" de "lo he cerrado yo"**. Tal cual se creó,
cerrar Docker a mano era inútil: lo relanzaba en menos de 5 minutos (verificado en el log:
`10:22:46 Docker Desktop no esta en ejecucion -> relanzando`).

```bat
scripts\watchdog_pausa.bat        REM deja de tocar Docker (crea logs\watchdog.pausa)
scripts\watchdog_reanudar.bat     REM vuelve a vigilar (borra el marcador)
```

Es un marcador **explícito** en vez de que el script adivine la intención: predecible sobre
mágico. Mientras esté pausado lo anota UNA vez en el log, no cada 5 minutos.

### La ventana que aparecía cada 5 minutos

`-WindowStyle Hidden` **no basta**: `powershell.exe` crea la consola y la oculta después, así
que con `LogonType=Interactive` se ve un destello en el escritorio en cada pasada. Dos arreglos
posibles:

```powershell
# A) sin admin — lanzar via conhost --headless (el que está aplicado)
$s = "C:\ruta\al\repo\scripts\docker_watchdog.ps1"
$a = New-ScheduledTaskAction -Execute "conhost.exe" `
     -Argument "--headless powershell.exe -ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File `"$s`""
Set-ScheduledTask -TaskName "MotorDockerWatchdog" -Action $a

# B) con admin — LogonType S4U (corre en segundo plano, sin sesión gráfica)
$p = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
Set-ScheduledTask -TaskName "MotorDockerWatchdog" -Principal $p
```

**Creación original de la tarea** (referencia; ya creada):

```bat
schtasks /Create /SC MINUTE /MO 5 /TN "MotorDockerWatchdog" ^
  /TR "powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File \"C:\ruta\al\repo\scripts\docker_watchdog.ps1\""
```

Borrar: `schtasks /Delete /TN "MotorDockerWatchdog" /F`
Ver estado: `Get-ScheduledTask -TaskName "Motor*" | Get-ScheduledTaskInfo`

Registra sus acciones en `logs/docker_watchdog.log` (no ensucia el log en el caso sano).
Nota: un resultado `3221225786` (`0xC000013A`) en la tarea significa "consola cerrada a
mano", no un fallo del script.

> **Camino de fondo (pendiente, evaluar con calma):** migrar a **WSL2 + Docker Engine
> nativo** (sin Docker Desktop) elimina la pieza que falla, en vez de vigilarla.
