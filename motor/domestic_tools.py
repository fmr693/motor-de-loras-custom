"""
motor.domestic_tools
====================
Seis herramientas domésticas reales para el agente personal.

Todas las operaciones destructivas incluyen un parámetro ``dry_run=True``
que por defecto **solo describe** lo que haría sin ejecutar nada.
Pasa ``dry_run=False`` para ejecutar de verdad.

Herramientas disponibles
------------------------
  file_organize   — organizar/mover archivos con reglas
  email_filter    — filtrar y gestionar correo IMAP
  calendar_get    — leer eventos del calendario (.ics o Outlook/Google)
  note_save       — guardar una nota en texto plano
  search_files    — buscar texto en archivos locales
  process_run     — ejecutar procesos de una whitelist autorizada

Uso con el agente
-----------------
  from motor.agent import LoRAAgent
  from motor.domestic_tools import DOMESTIC_TOOLS

  agent = LoRAAgent(infer_fn=my_infer, tools=DOMESTIC_TOOLS)
  result = agent.run("Organiza las facturas del escritorio")
"""

from __future__ import annotations

import imaplib
import email
import os
import re
import shutil
import subprocess
import textwrap
from datetime import datetime, timedelta
from email.header import decode_header
from pathlib import Path
from typing import Optional

from motor.agent import Tool


# ---------------------------------------------------------------------------
# Detección de plataforma
# ---------------------------------------------------------------------------

_IS_WINDOWS = os.name == "nt"
_IS_LINUX   = os.name == "posix"

# ---------------------------------------------------------------------------
# Seguridad compartida
# ---------------------------------------------------------------------------

def _build_safe_roots() -> list[Path]:
    """
    Construye las raíces seguras de forma portable (Windows / Linux / Docker).
    En Linux/Docker, el home puede ser /root o /home/user.
    Solo incluye rutas que pueden existir en la plataforma actual.
    """
    home = Path.home()
    candidates = [
        # Subdirectorios del home (NO el home en sí para evitar que AppData, tmp, etc. sean seguros)
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
        home / "Pictures",
        home / "Music",
        home / "Videos",
        home / "Notes",
        # Español (Windows y Linux con locale es)
        home / "Escritorio",
        home / "Documentos",
        home / "Descargas",
        home / "Imágenes",
        home / "Música",
        home / "Notas",
    ]
    if _IS_LINUX:
        # En Docker/Linux incluir /tmp y /app/data como zonas seguras
        candidates += [Path("/tmp"), Path("/app/data"), Path("/app/logs")]
    return candidates

# Rutas raíz donde se permiten operaciones de escritura/movimiento.
_SAFE_WRITE_ROOTS: list[Path] = _build_safe_roots()

# Raíces pre-resueltas (evita re-resolver en cada llamada a _is_safe_path).
_SAFE_ROOTS_RESOLVED: list[str] = [
    str(r.expanduser().resolve()) for r in _SAFE_WRITE_ROOTS
]

# Procesos que el agente puede ejecutar (solo estos, ningún otro).
# Separados por plataforma para evitar exponer comandos inexistentes.
_PROCESS_WHITELIST: dict[str, str] = {
    # Universales
    "python":   "python",
    "python3":  "python3",
    "ping":     "ping",
    "find":     "find",
    "ls":       "ls" if _IS_LINUX else "dir",
    # Windows-only
    **({"notepad":    "notepad.exe",
        "explorador": "explorer.exe",
        "calculadora":"calc.exe",
        "ipconfig":   "ipconfig",
        "dir":        "dir",
    } if _IS_WINDOWS else {}),
    # Linux-only
    **({"ifconfig":  "ifconfig",
        "ip":        "ip",
        "cat":       "cat",
        "grep":      "grep",
    } if _IS_LINUX else {}),
}


def _is_safe_path(path: Path) -> bool:
    """True si la ruta está dentro de alguna raíz segura (la raíz no tiene que existir)."""
    resolved_str = str(path.expanduser().resolve())
    return any(
        resolved_str == root or resolved_str.startswith(root + os.sep)
        for root in _SAFE_ROOTS_RESOLVED
    )


def _expand(p: str) -> Path:
    return Path(p).expanduser().resolve()


# Alias bilingüe de carpetas comunes (inglés ↔ español y variantes de mayúsculas)
_DIR_ALIASES: dict[str, list[str]] = {
    "desktop":    ["Desktop", "Escritorio", "escritorio"],
    "escritorio": ["Escritorio", "Desktop", "escritorio"],
    "downloads":  ["Downloads", "Descargas", "downloads"],
    "descargas":  ["Descargas", "Downloads", "descargas"],
    "documents":  ["Documents", "Documentos", "documents"],
    "documentos": ["Documentos", "Documents", "documentos"],
    "pictures":   ["Pictures", "Imágenes", "Images"],
    "imágenes":   ["Imágenes", "Pictures", "Images"],
    "music":      ["Music", "Música", "musica"],
    "música":     ["Música", "Music"],
    "videos":     ["Videos", "Vídeos", "video"],
    "vídeos":     ["Vídeos", "Videos"],
    "notes":      ["Notes", "Notas", "Notas"],
    "notas":      ["Notas", "Notes"],
}


def _fuzzy_path(p: str) -> Path:
    """
    Expande la ruta y, si no existe, intenta variantes bilingües del último
    componente de la ruta (ej. "Downloads" → "Descargas").
    Devuelve el primer Path existente; si ninguno existe, devuelve el original.
    """
    base = _expand(p)
    if base.exists():
        return base
    # Intentar variantes del componente final
    parent = base.parent
    name = base.name.lower()
    for variant in _DIR_ALIASES.get(name, []):
        candidate = parent / variant
        candidate_exp = candidate.expanduser().resolve() if not candidate.is_absolute() else candidate
        if candidate_exp.exists():
            return candidate_exp
    return base  # devolver original aunque no exista (el caller gestiona el error)


# ---------------------------------------------------------------------------
# 1. file_organize
# ---------------------------------------------------------------------------

def file_organize(
    files:    list[str],
    dest:     str,
    dry_run:  bool = True,
) -> str:
    """
    Mueve una lista de archivos a una carpeta destino.

    Parámetros
    ----------
    files   : lista de rutas absolutas o relativas (soporte ~ y ~user)
    dest    : carpeta destino (se crea si no existe)
    dry_run : True → solo describe la operación sin ejecutarla (por defecto)

    Devuelve una descripción de cada acción realizada (o que se realizaría).
    """
    dest_path = _fuzzy_path(dest)
    dest_warning = "" if str(dest_path) == str(_expand(dest)) else f"[Aviso]: destino ajustado a '{dest_path}'.\n"

    if not dry_run and not _is_safe_path(dest_path):
        return (
            f"[Bloqueado]: el destino '{dest}' está fuera de las rutas "
            f"seguras permitidas. Usa una carpeta dentro de "
            f"Escritorio, Documentos, Descargas, Notas o similares."
        )

    lines: list[str] = []
    prefix = "[dry_run] " if dry_run else ""

    if not dry_run:
        dest_path.mkdir(parents=True, exist_ok=True)

    moved = 0
    errors = 0
    for f in files:
        src = _expand(f)
        if not src.exists():
            lines.append(f"  {prefix}✗  No encontrado: {f}")
            errors += 1
            continue
        if not src.is_file():
            lines.append(f"  {prefix}✗  No es un archivo: {f}")
            errors += 1
            continue
        target = dest_path / src.name
        if target.exists():
            lines.append(f"  {prefix}⚠  Ya existe en destino: {src.name} (se omite)")
            continue
        if dry_run:
            lines.append(f"  {prefix}→  {src.name}  ({src.stat().st_size:,} B)  →  {dest_path}/")
        else:
            if not _is_safe_path(src):
                lines.append(f"  ✗  Ruta origen no permitida: {f}")
                errors += 1
                continue
            shutil.move(str(src), str(target))
            lines.append(f"  ✓  {src.name}  →  {dest_path}/")
            moved += 1

    summary = (
        f"{'[dry_run] ' if dry_run else ''}"
        f"file_organize: {len(files)} archivos procesados, "
        f"{moved if not dry_run else len([l for l in lines if '→' in l])} a mover, "
        f"{errors} errores."
    )
    if dry_run:
        summary += " Pasa dry_run=False para ejecutar."

    return dest_warning + summary + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. email_filter
# ---------------------------------------------------------------------------

def email_filter(
    server:   str,
    user:     str,
    password: str,
    folder:   str   = "INBOX",
    sender:   Optional[str] = None,
    subject:  Optional[str] = None,
    mode:     str   = "list",
    limit:    int   = 20,
    dry_run:  bool  = True,
) -> str:
    """
    Filtra y gestiona mensajes en una cuenta IMAP.

    Parámetros
    ----------
    server   : servidor IMAP (ej. "imap.gmail.com")
    user     : dirección de correo
    password : contraseña o token de aplicación  (nunca se loguea)
    folder   : carpeta IMAP (default: INBOX)
    sender   : filtrar por remitente (parcial, case-insensitive)
    subject  : filtrar por asunto (parcial, case-insensitive)
    mode     : "list" | "move" | "delete" | "mark_spam"
    limit    : máximo de mensajes a procesar (default: 20)
    dry_run  : True → solo lista, no modifica nada (por defecto)

    Devuelve un resumen de los mensajes encontrados/modificados.
    """
    if mode != "list" and dry_run:
        note = f"[dry_run] email_filter modo='{mode}' — no se modificará nada. "
    else:
        note = ""

    try:
        conn = imaplib.IMAP4_SSL(server)
    except Exception as e:
        return f"[Error de conexión IMAP]: {e}"

    try:
        conn.login(user, password)
    except imaplib.IMAP4.error as e:
        return f"[Error de autenticación IMAP]: {e}"

    try:
        conn.select(folder, readonly=(dry_run or mode == "list"))

        # Construir criterio de búsqueda
        # Sanitizar sender/subject para prevenir IMAP injection
        def _imap_escape(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"')

        criteria: list[str] = ["ALL"]
        if sender:
            criteria = [f'FROM "{_imap_escape(sender)}"']
        if subject:
            safe_subj = f'SUBJECT "{_imap_escape(subject)}"'
            criteria = criteria + [safe_subj] if criteria != ["ALL"] else [safe_subj]

        search_str = " ".join(criteria) if len(criteria) == 1 else f"({' '.join(criteria)})"
        _, data = conn.search(None, search_str)
        msg_ids = data[0].split() if data[0] else []
        msg_ids = msg_ids[-limit:]  # los más recientes primero

        if not msg_ids:
            return note + f"No se encontraron mensajes en '{folder}'" + (
                f" de '{sender}'" if sender else ""
            ) + (f" con asunto '{subject}'" if subject else "") + "."

        def _decode_header(val: str) -> str:
            parts = decode_header(val or "")
            out = []
            for part, enc in parts:
                if isinstance(part, bytes):
                    out.append(part.decode(enc or "utf-8", errors="replace"))
                else:
                    out.append(part)
            return "".join(out)

        lines: list[str] = []
        for uid in msg_ids[-10:]:  # mostrar máx 10 en el resumen
            _, msg_data = conn.fetch(uid, "(RFC822.HEADER)")
            raw = msg_data[0][1] if msg_data and msg_data[0] else b""
            msg = email.message_from_bytes(raw)

            from_h    = _decode_header(msg.get("From", "(sin remitente)"))
            subject_h = _decode_header(msg.get("Subject", "(sin asunto)"))
            date_h    = msg.get("Date", "")[:16]
            lines.append(f"  [{uid.decode()}] {date_h}  De: {from_h[:40]}  Asunto: {subject_h[:50]}")

        if len(msg_ids) > 10:
            lines.append(f"  ... y {len(msg_ids) - 10} mensajes más")

        summary = (
            f"{note}email_filter: {len(msg_ids)} mensajes encontrados en '{folder}'"
        )
        if mode != "list" and not dry_run:
            summary += f" — acción '{mode}' pendiente de implementación."
        elif mode != "list":
            summary += f" — acción '{mode}' se ejecutaría sobre {len(msg_ids)} mensajes."

        return summary + "\n" + "\n".join(lines)

    except Exception as e:
        return f"[Error IMAP]: {e}"
    finally:
        try:
            conn.logout()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 3. calendar_get
# ---------------------------------------------------------------------------

def calendar_get(date_range: str = "today") -> str:
    """
    Lee eventos del calendario local.

    Parámetros
    ----------
    date_range : "today" | "week" | "month" | "YYYY-MM-DD" | "YYYY-MM-DD/YYYY-MM-DD"

    Busca archivos .ics en las carpetas habituales del sistema.
    Si no encuentra ninguno, intenta el calendario de Outlook/Google vía
    registro del SO (no implementado en esta versión: devuelve mensaje claro).
    """
    today = datetime.now().date()

    if date_range == "today":
        start = end = today
    elif date_range == "week":
        start = today
        end   = today + timedelta(days=6)
    elif date_range == "month":
        start = today
        end   = today + timedelta(days=29)
    elif "/" in date_range:
        parts = date_range.split("/", 1)
        try:
            start = datetime.strptime(parts[0], "%Y-%m-%d").date()
            end   = datetime.strptime(parts[1], "%Y-%m-%d").date()
        except ValueError:
            return f"[Error]: formato de fecha no reconocido '{date_range}'. Usa YYYY-MM-DD/YYYY-MM-DD"
    else:
        try:
            start = end = datetime.strptime(date_range, "%Y-%m-%d").date()
        except ValueError:
            return f"[Error]: formato no reconocido '{date_range}'. Usa: today|week|month|YYYY-MM-DD"

    # Buscar archivos .ics en rutas habituales
    ics_roots = [
        Path.home() / "Calendar",
        Path.home() / "Calendarios",
        Path.home() / "Documents" / "Calendar",
        Path.home() / "Documentos" / "Calendario",
        Path.home() / "AppData" / "Local" / "Packages",   # Outlook new app
    ]
    ics_files: list[Path] = []
    for root in ics_roots:
        if root.exists():
            ics_files.extend(root.rglob("*.ics"))

    if not ics_files:
        return (
            f"calendar_get ({date_range}): No se encontraron archivos .ics en las "
            f"rutas habituales ({', '.join(str(r) for r in ics_roots[:3])}, …). "
            f"Para usar Google Calendar o Outlook, exporta el calendario como .ics "
            f"y guárdalo en ~/Calendarios/."
        )

    try:
        import icalendar  # type: ignore
    except ImportError:
        return (
            "calendar_get: se encontraron archivos .ics pero el paquete "
            "'icalendar' no está instalado. Ejecuta: pip install icalendar"
        )

    events: list[tuple[datetime, str, str]] = []
    for ics_path in ics_files[:5]:  # máx 5 archivos
        try:
            cal = icalendar.Calendar.from_ical(ics_path.read_bytes())
            for component in cal.walk():
                if component.name != "VEVENT":
                    continue
                dtstart = component.get("DTSTART")
                if dtstart is None:
                    continue
                ev_date = dtstart.dt
                if hasattr(ev_date, "date"):
                    ev_date = ev_date.date()
                if start <= ev_date <= end:
                    summary_text = str(component.get("SUMMARY", "(sin título)"))
                    location     = str(component.get("LOCATION", ""))
                    events.append((ev_date, summary_text, location))
        except Exception:
            continue

    if not events:
        return f"calendar_get ({date_range}): No se encontraron eventos entre {start} y {end}."

    events.sort(key=lambda x: x[0])
    lines = [f"  {d}  {s}" + (f"  @{l}" if l else "") for d, s, l in events]
    return (
        f"calendar_get ({date_range}): {len(events)} evento(s) entre {start} y {end}.\n"
        + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# 4. note_save
# ---------------------------------------------------------------------------

def note_save(
    title:  str,
    body:   str,
    folder: str = "~/Notas",
) -> str:
    """
    Guarda una nota como archivo de texto plano.

    Parámetros
    ----------
    title  : título de la nota (se usa como nombre de archivo)
    body   : contenido de la nota
    folder : carpeta donde guardar (default: ~/Notas)

    El archivo se nombra  <timestamp>_<titulo_sanitizado>.txt
    Si ya existe, se añade un sufijo numérico para no sobreescribir.
    """
    folder_path = _expand(folder)

    if not _is_safe_path(folder_path):
        return (
            f"[Bloqueado]: la carpeta '{folder}' está fuera de las rutas "
            f"seguras. Usa ~/Notas, ~/Documentos o similares."
        )

    folder_path.mkdir(parents=True, exist_ok=True)

    # Sanitizar el título para usarlo como nombre de archivo
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title).strip()[:80]
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"{timestamp}_{safe_title}.txt"
    target     = folder_path / filename

    # Evitar sobreescritura (no debería ocurrir con timestamp, pero por si acaso)
    counter = 1
    while target.exists():
        target = folder_path / f"{timestamp}_{safe_title}_{counter}.txt"
        counter += 1

    content = f"{title}\n{'=' * len(title)}\n{datetime.now():%Y-%m-%d %H:%M}\n\n{body}\n"
    target.write_text(content, encoding="utf-8")

    return (
        f"note_save: nota guardada en '{target}' "
        f"({len(content):,} caracteres)."
    )


# ---------------------------------------------------------------------------
# 5. search_files
# ---------------------------------------------------------------------------

def search_files(
    query:      str,
    path:       str              = "~/Desktop",
    extensions: Optional[list[str]] = None,
    max_results: int             = 20,
    context_lines: int           = 1,
) -> str:
    """
    Busca texto en archivos locales (búsqueda de contenido, no de nombre).

    Parámetros
    ----------
    query          : texto a buscar (literal, case-insensitive)
    path           : carpeta raíz donde buscar (default: ~/Desktop)
    extensions     : lista de extensiones a incluir, ej. [".txt", ".md", ".py"]
                     Si es None, busca en todos los archivos de texto (<1 MB)
    max_results    : máximo de coincidencias a devolver (default: 20)
    context_lines  : líneas de contexto alrededor de cada coincidencia (default: 1)

    Devuelve las coincidencias con su ruta y número de línea.
    """
    root = _fuzzy_path(path)
    if not root.exists():
        return f"[Error]: la ruta '{path}' no existe."
    if not root.is_dir():
        return f"[Error]: '{path}' no es un directorio."
    warning = "" if str(root) == str(_expand(path)) else f"[Aviso]: ruta ajustada a '{root}'.\n"

    allowed_ext = {e.lower() if e.startswith(".") else f".{e.lower()}"
                   for e in extensions} if extensions else None

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    results: list[str] = []
    files_checked = 0
    files_matched = 0

    for fpath in root.rglob("*"):
        if not fpath.is_file():
            continue
        if allowed_ext and fpath.suffix.lower() not in allowed_ext:
            continue
        if fpath.stat().st_size > 1_000_000:  # omitir archivos >1 MB
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        files_checked += 1
        file_matches = 0
        lines = text.splitlines()

        for i, line in enumerate(lines):
            if pattern.search(line):
                if len(results) >= max_results:
                    break
                # Contexto
                start = max(0, i - context_lines)
                end   = min(len(lines), i + context_lines + 1)
                ctx   = "\n".join(
                    f"    {'→' if j == i else ' '} {j+1:4d}: {lines[j]}"
                    for j in range(start, end)
                )
                results.append(f"  {fpath}  (línea {i+1})\n{ctx}")
                file_matches += 1

        if file_matches > 0:
            files_matched += 1

    if not results:
        return (
            warning +
            f"search_files: ninguna coincidencia para '{query}' "
            f"en '{root}' ({files_checked} archivos revisados)."
        )

    truncated = len(results) >= max_results
    header = (
        warning +
        f"search_files: {len(results)}{'+ ' if truncated else ' '}"
        f"coincidencia(s) en {files_matched} archivo(s) "
        f"({files_checked} revisados).\n"
    )
    return header + "\n".join(results) + (
        f"\n  [truncado a {max_results} resultados]" if truncated else ""
    )


# ---------------------------------------------------------------------------
# 6. process_run
# ---------------------------------------------------------------------------

def process_run(
    name:    str,
    args:    Optional[list[str]] = None,
    dry_run: bool = True,
    timeout: int  = 10,
) -> str:
    """
    Ejecuta un proceso de la whitelist de procesos autorizados.

    Parámetros
    ----------
    name    : nombre del proceso (ver lista en _PROCESS_WHITELIST)
    args    : argumentos adicionales como lista de strings
    dry_run : True → solo describe el comando sin ejecutarlo (por defecto)
    timeout : segundos máximos de espera para la salida (default: 10)

    Solo se pueden ejecutar procesos en la whitelist.
    Devuelve la salida estándar del proceso o una descripción (dry_run).
    """
    key = name.lower().strip()
    if key not in _PROCESS_WHITELIST:
        available = ", ".join(sorted(_PROCESS_WHITELIST.keys()))
        return (
            f"[Bloqueado]: '{name}' no está en la whitelist de procesos permitidos.\n"
            f"Procesos disponibles: {available}"
        )

    executable = _PROCESS_WHITELIST[key]
    cmd = [executable] + (args or [])
    cmd_str = " ".join(cmd)

    if dry_run:
        return (
            f"[dry_run] process_run: ejecutaría → {cmd_str}\n"
            f"Pasa dry_run=False para ejecutar."
        )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip() or result.stderr.strip() or "(sin salida)"
        # Limitar la salida para no desbordar el contexto del agente
        if len(output) > 2000:
            output = output[:2000] + "\n[...truncado]"
        return (
            f"process_run: '{cmd_str}' → código {result.returncode}\n"
            f"{output}"
        )
    except subprocess.TimeoutExpired:
        return f"[Timeout]: '{cmd_str}' no respondió en {timeout}s."
    except FileNotFoundError:
        return f"[Error]: ejecutable '{executable}' no encontrado en el PATH."
    except Exception as e:
        return f"[Error]: {e}"


# ---------------------------------------------------------------------------
# Registro de herramientas para LoRAAgent
# ---------------------------------------------------------------------------

DOMESTIC_TOOLS: list[Tool] = [
    Tool(
        name="file_organize",
        description=(
            "Mueve o reorganiza archivos a una carpeta destino. "
            "Usa dry_run=True (defecto) para ver qué haría sin ejecutar. "
            "Pasa dry_run=False solo cuando el usuario confirme la operación."
        ),
        params_doc=textwrap.dedent("""\
            files   (list[str]): rutas de los archivos a mover
            dest    (str): carpeta destino (se crea si no existe)
            dry_run (bool): True=solo describe, False=ejecuta (default: True)
        """),
        fn=file_organize,
        input_schema={
            "type": "object",
            "properties": {
                "files":   {"type": "array", "items": {"type": "string"}},
                "dest":    {"type": "string"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["files", "dest"],
        },
    ),
    Tool(
        name="email_filter",
        description=(
            "Lista o gestiona correos en una cuenta IMAP. "
            "Filtros opcionales por remitente y asunto. "
            "Modos: 'list' (solo lee), 'move', 'delete', 'mark_spam'. "
            "Con dry_run=True (defecto) nunca modifica nada."
        ),
        params_doc=textwrap.dedent("""\
            server   (str): servidor IMAP, ej. "imap.gmail.com"
            user     (str): dirección de correo
            password (str): contraseña o token de aplicación
            folder   (str): carpeta IMAP (default: "INBOX")
            sender   (str): filtrar por remitente (opcional)
            subject  (str): filtrar por asunto (opcional)
            mode     (str): "list"|"move"|"delete"|"mark_spam" (default: "list")
            limit    (int): máx. mensajes a procesar (default: 20)
            dry_run  (bool): True=solo describe, False=ejecuta (default: True)
        """),
        fn=email_filter,
        input_schema={
            "type": "object",
            "properties": {
                "server":   {"type": "string"},
                "user":     {"type": "string"},
                "password": {"type": "string"},
                "folder":   {"type": "string"},
                "sender":   {"type": "string"},
                "subject":  {"type": "string"},
                "mode":     {"type": "string", "enum": ["list", "move", "delete", "mark_spam"]},
                "limit":    {"type": "integer"},
                "dry_run":  {"type": "boolean"},
            },
            "required": ["server", "user", "password"],
        },
    ),
    Tool(
        name="calendar_get",
        description=(
            "Lee eventos del calendario local desde archivos .ics. "
            "Rango de fechas: 'today', 'week', 'month', 'YYYY-MM-DD' o 'YYYY-MM-DD/YYYY-MM-DD'."
        ),
        params_doc=textwrap.dedent("""\
            date_range (str): "today"|"week"|"month"|"YYYY-MM-DD"|"YYYY-MM-DD/YYYY-MM-DD"
        """),
        fn=calendar_get,
        input_schema={
            "type": "object",
            "properties": {
                "date_range": {"type": "string"},
            },
            "required": ["date_range"],
        },
    ),
    Tool(
        name="note_save",
        description=(
            "Guarda una nota en texto plano en ~/Notas (u otra carpeta segura). "
            "El archivo se nombra con timestamp + título para evitar sobreescrituras."
        ),
        params_doc=textwrap.dedent("""\
            title  (str): título de la nota
            body   (str): contenido de la nota
            folder (str): carpeta donde guardar (default: "~/Notas")
        """),
        fn=note_save,
        input_schema={
            "type": "object",
            "properties": {
                "title":  {"type": "string"},
                "body":   {"type": "string"},
                "folder": {"type": "string"},
            },
            "required": ["title", "body"],
        },
    ),
    Tool(
        name="search_files",
        description=(
            "Busca texto dentro de archivos locales. "
            "Devuelve ruta, número de línea y contexto de cada coincidencia. "
            "Filtra por extensión con el parámetro 'extensions'."
        ),
        params_doc=textwrap.dedent("""\
            query          (str): texto a buscar (literal, sin regex)
            path           (str): carpeta raíz donde buscar (default: "~/Desktop")
            extensions     (list[str]): extensiones a incluir, ej. [".txt", ".md"] (default: todas)
            max_results    (int): máximo de resultados (default: 20)
            context_lines  (int): líneas de contexto (default: 1)
        """),
        fn=search_files,
        input_schema={
            "type": "object",
            "properties": {
                "query":         {"type": "string"},
                "path":          {"type": "string"},
                "extensions":    {"type": "array", "items": {"type": "string"}},
                "max_results":   {"type": "integer"},
                "context_lines": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="process_run",
        description=(
            "Ejecuta un proceso de la lista autorizada. "
            "Procesos disponibles: notepad, explorador, calculadora, python, "
            "ping, ipconfig, ifconfig, ls, dir. "
            "Con dry_run=True (defecto) solo muestra el comando sin ejecutar."
        ),
        params_doc=textwrap.dedent("""\
            name    (str): nombre del proceso (de la whitelist)
            args    (list[str]): argumentos adicionales (opcional)
            dry_run (bool): True=solo describe, False=ejecuta (default: True)
            timeout (int): segundos de espera máxima (default: 10)
        """),
        fn=process_run,
        input_schema={
            "type": "object",
            "properties": {
                "name":    {"type": "string"},
                "args":    {"type": "array", "items": {"type": "string"}},
                "dry_run": {"type": "boolean"},
                "timeout": {"type": "integer"},
            },
            "required": ["name"],
        },
    ),
]
