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
  --dest "C:/Users/Felipe/OneDrive/backups_motor" \
  --memory-dir "C:/Users/Felipe/.claude/projects/c--Users-Felipe-Desktop-Proyecto-motor-de-loras-custom/memory"
```

**Tarea programada semanal** (domingos 20:00):

```bat
schtasks /Create /SC WEEKLY /D SUN /ST 20:00 /TN "MotorBackupActivo" ^
  /TR "python \"C:\Users\Felipe\Desktop\Proyecto\motor-de-loras-custom\scripts\backup_activo.py\" --memory-dir \"C:\Users\Felipe\.claude\projects\c--Users-Felipe-Desktop-Proyecto-motor-de-loras-custom\memory\""
```

Borrar: `schtasks /Delete /TN "MotorBackupActivo" /F`

> **AVISO:** el destino por defecto está en el **mismo disco** que el proyecto — no
> protege contra un disco muerto. Para protección real, usa `--dest` en otro disco o
> una carpeta sincronizada a la nube (OneDrive/Drive).

---

## `docker_watchdog.ps1` — vigilante de Docker

Docker Desktop se cayó **dos veces** en una sola sesión de pruebas de estrés: es el
eslabón frágil del stack soberano en Windows. Este watchdog comprueba el engine y, si
no responde, relanza Docker Desktop. Los contenedores (`restart: unless-stopped`)
vuelven solos en cuanto el engine está arriba.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\docker_watchdog.ps1
```

**Tarea programada cada 5 minutos:**

```bat
schtasks /Create /SC MINUTE /MO 5 /TN "MotorDockerWatchdog" ^
  /TR "powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File \"C:\Users\Felipe\Desktop\Proyecto\motor-de-loras-custom\scripts\docker_watchdog.ps1\""
```

Borrar: `schtasks /Delete /TN "MotorDockerWatchdog" /F`

Registra sus acciones en `logs/docker_watchdog.log` (no ensucia el log en el caso sano).

> **Camino de fondo (pendiente, evaluar con calma):** migrar a **WSL2 + Docker Engine
> nativo** (sin Docker Desktop) elimina la pieza que falla, en vez de vigilarla.
