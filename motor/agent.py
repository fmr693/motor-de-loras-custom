"""
motor.agent
===========
Agente ReAct (Reason + Act) para la Fábrica de LoRAs.

El agente recibe una tarea en lenguaje natural, razona paso a paso y usa
herramientas para resolverla.  Cada ciclo: Thought → Action → Observation
hasta llegar a Final Answer o agotar el máximo de pasos.

Herramientas incluidas
----------------------
  read_file  — lee un archivo de texto (máx. 50 KB / 200 líneas)
  list_dir   — lista el contenido de un directorio
  shell      — ejecuta comandos de solo lectura (ls, grep, nvidia-smi…)
               Comandos destructivos están bloqueados.
  http_get   — hace GET a una URL y devuelve el cuerpo

Uso directo (sin servidor)
--------------------------
  from motor.agent import LoRAAgent, DEFAULT_TOOLS
  from motor.server import _infer   # ya cargado

  agent = LoRAAgent(infer_fn=_infer)
  result = agent.run("¿Cuántos ejemplos tiene datasets/titanic.jsonl?")
  print(result.answer)
  for step in result.steps:
      print(step.to_dict())
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utilidades de parsing
# ---------------------------------------------------------------------------

def _extract_json_object(text: str, start: int) -> Optional[str]:
    """
    Extrae el primer objeto JSON completo desde la posición ``start``
    (que debe apuntar al carácter ``{`` inicial).

    A diferencia de un simple regex, respeta llaves anidadas y llaves
    dentro de cadenas (ej. comandos con awk/sed que contienen '{...}').

    Devuelve la subcadena ``{...}`` o None si no puede extraerla.
    """
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Sanitización de outputs del LLM
# ---------------------------------------------------------------------------

# Patrones que delatan un template sin rellenar (solo los inequívocos)
_SANITIZE_TEMPLATE_RE = re.compile(
    r"\{\{[^}]*\}\}"           # {{algo}} — Jinja2 / doble llave
    r"|\{[A-Z_]{2,}\}"         # {MAYUSCULAS} — placeholder tipo {NAME}
    r"|\[[A-Z_]{2,}\]"         # [MAYUSCULAS] — placeholder tipo [NAME]
    r"|\[placeholder\]"        # literal [placeholder]
    r"|<[A-Z_]{2,}>",          # <MAYUSCULAS> — placeholder HTML-like
    re.IGNORECASE,
)

# Frases que delatan alucinación o respuesta vacía
_SANITIZE_HALLUCINATION_PHRASES = [
    "no tengo información",
    "no dispongo",
    "no puedo responder",
    "como modelo de lenguaje",
    "lo siento",
    "no está disponible",
    "no se encontró",
    "no se puede determinar",
    "no se puede saber",
    "no se proporcionó",
    "información no disponible",
    "respuesta no encontrada",
]


def _sanitize_output(text: str) -> Optional[str]:
    """
    Revisa un texto (Final Answer u observación) en busca de templates sin
    rellenar, placeholders, o señales claras de alucinación.

    Devuelve None si el texto es válido, o un string con el motivo del
    rechazo si encuentra algún problema.
    """
    if not text or not text.strip():
        return "[Sanitización]: Respuesta vacía o nula."

    # 1) Templates sin rellenar (solo patrones inequívocos)
    m = _SANITIZE_TEMPLATE_RE.search(text)
    if m:
        return (
            "[Sanitización]: Se detectó un template sin rellenar: "
            f"'{m.group()}'"
        )

    # 2) Frases de alucinación
    lowered = text.lower()
    for phrase in _SANITIZE_HALLUCINATION_PHRASES:
        if phrase in lowered:
            return (
                "[Sanitización]: Se detectó posible alucinación o "
                f"respuesta inválida: '{phrase}'"
            )

    # 3) Respuesta demasiado corta para ser útil
    if len(text.strip()) < 3:
        return "[Sanitización]: Respuesta demasiado corta."

    return None


def _sanitize_observation(text: str) -> Optional[str]:
    """
    Versión más laxa de sanitización para observaciones de herramientas.
    Solo rechaza: vacío total, error interno no controlado, o template
    doble-llave ({{...}}) que es inequívoco.
    """
    if not text or not text.strip():
        return "[Sanitización]: Observación vacía."

    # Solo doble llave es inequívoca en output de herramientas
    m = re.search(r"\{\{[^}]*\}\}", text)
    if m:
        return (
            "[Sanitización]: Observación contiene template sin rellenar: "
            f"'{m.group()}'"
        )

    return None


# ---------------------------------------------------------------------------
# Herramientas
# ---------------------------------------------------------------------------

class Tool:
    """Envuelve una función Python como herramienta del agente."""

    def __init__(
        self,
        name:         str,
        description:  str,
        params_doc:   str,
        fn:           Callable[..., str],
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name         = name
        self.description  = description
        self.params_doc   = params_doc
        self._fn          = fn
        self.input_schema = input_schema
        """
        JSON Schema que describe los parámetros de la herramienta.
        Cuando está presente, el agente usará inferencia en dos fases:
          Fase 1 — texto libre: el modelo razona y elige la acción.
          Fase 2 — JSON Schema constrained: solo genera el Action Input
                   válido, sin poder alucinar campos ni tipos.
        Ejemplo mínimo (file_organize)::

            {
              "type": "object",
              "properties": {
                "files": {"type": "array", "items": {"type": "string"}},
                "dest":  {"type": "string"},
                "dry_run": {"type": "boolean"}
              },
              "required": ["files", "dest"]
            }

        Si es None (por defecto), se usa el comportamiento clásico: el
        modelo genera JSON libre y se aplica el parser de 5 niveles.
        """

    def __call__(self, **kwargs: Any) -> str:
        try:
            return str(self._fn(**kwargs))
        except TypeError as e:
            return f"[Error de parámetros en '{self.name}']: {e}"
        except Exception as e:
            return f"[Error en '{self.name}']: {e}"


# --- read_file ---------------------------------------------------------------

def _tool_read_file(path: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Archivo no encontrado: {path}"
    if not p.is_file():
        return f"No es un archivo: {path}"
    size = p.stat().st_size
    if size > 50_000:
        lines = p.read_text(errors="replace").splitlines()[:200]
        return "\n".join(lines) + f"\n\n[...truncado — {size:,} bytes en total]"
    return p.read_text(errors="replace")


# --- list_dir ----------------------------------------------------------------

def _tool_list_dir(path: str = ".") -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Ruta no encontrada: {path}"
    if not p.is_dir():
        return f"No es un directorio: {path}"
    items = []
    for item in sorted(p.iterdir()):
        if item.is_dir():
            items.append(f"  {item.name}/")
        else:
            items.append(f"  {item.name}  ({item.stat().st_size:,} B)")
    return f"{p}/\n" + ("\n".join(items) if items else "  (vacío)")


# --- shell -------------------------------------------------------------------

# Patrones de comandos peligrosos bloqueados
_SHELL_BLOCKED = re.compile(
    r"(^|\s|;|&&|\|)("
    r"rm\b|rmdir\b|dd\b|mkfs\b|fdisk\b|shred\b|truncate\b"
    r"|sudo\b|su\b|chmod\b|chown\b|passwd\b|useradd\b|userdel\b"
    r"|shutdown\b|reboot\b|halt\b|poweroff\b"
    r"|apt\b|apt-get\b|yum\b|dnf\b|snap\b"
    r"|pip install\b|pip3 install\b|python\s+-m\s+pip\b|python3\s+-m\s+pip\b"
    r"|wget\b|curl\b"
    r"|>\s*/dev/|>\s*/etc/|>\s*/boot/"
    r")",
    re.IGNORECASE,
)

def _tool_shell(command: str) -> str:
    if _SHELL_BLOCKED.search(command):
        return (
            "[Bloqueado]: ese comando no está permitido por seguridad.\n"
            "Solo se permiten comandos de lectura: ls, cat, grep, find, "
            "head, tail, wc, du, df, nvidia-smi, ps, python -c, etc."
        )
    try:
        result = subprocess.run(
            command,
            shell        = True,
            capture_output = True,
            text         = True,
            timeout      = 15,
            cwd          = Path.home() / "Proyecto_V3",
        )
        out = (result.stdout or "") + (result.stderr or "")
        if not out.strip():
            return "(sin salida)"
        return out[:3_000] + ("\n[...truncado]" if len(out) > 3_000 else "")
    except subprocess.TimeoutExpired:
        return "[Timeout]: el comando tardó más de 15 segundos."
    except Exception as e:
        return f"[Error de shell]: {e}"


# --- http_get ----------------------------------------------------------------

def _tool_http_get(url: str) -> str:
    import urllib.request
    import urllib.error
    # Solo permitir http/https
    if not url.startswith(("http://", "https://")):
        return "[Error]: solo se permiten URLs http:// o https://"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            content = r.read(10_000).decode("utf-8", errors="replace")
        return content
    except urllib.error.URLError as e:
        return f"[Error HTTP]: {e}"
    except Exception as e:
        return f"[Error]: {e}"


# --- Lista de herramientas por defecto --------------------------------------

DEFAULT_TOOLS: List[Tool] = [
    Tool(
        name        = "read_file",
        description = "Lee el contenido de un archivo de texto (máx. 200 líneas).",
        params_doc  = '{"path": "datasets/titanic.jsonl"}',
        fn          = _tool_read_file,
    ),
    Tool(
        name        = "list_dir",
        description = "Lista el contenido de un directorio con tamaños.",
        params_doc  = '{"path": "adapters/"}',
        fn          = _tool_list_dir,
    ),
    Tool(
        name        = "shell",
        description = (
            "Ejecuta un comando de shell de solo lectura. "
            "Permitido: ls, cat, grep, find, head, tail, wc, du, df, "
            "nvidia-smi, ps, python -c \"...\", etc. "
            "Bloqueado: rm, dd, wget, curl, pip install, sudo, etc."
        ),
        params_doc  = '{"command": "nvidia-smi --query-gpu=memory.used --format=csv"}',
        fn          = _tool_shell,
    ),
    Tool(
        name        = "http_get",
        description = "Hace una petición GET a una URL y devuelve el cuerpo (máx. 10 KB).",
        params_doc  = '{"url": "http://localhost:8000/health"}',
        fn          = _tool_http_get,
    ),
]


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    thought:      str
    action:       Optional[str]  = None
    action_input: Optional[dict] = None
    observation:  Optional[str]  = None
    is_final:     bool           = False
    final_answer: Optional[str]  = None

    def to_dict(self) -> dict:
        d: dict = {"thought": self.thought}
        if self.action:
            d["action"]       = self.action
            d["action_input"] = self.action_input
        if self.observation is not None:
            d["observation"] = self.observation
        if self.is_final:
            d["final_answer"] = self.final_answer
        return d


@dataclass
class AgentResult:
    answer:  str
    steps:   List[AgentStep] = field(default_factory=list)
    success: bool            = True

    def to_dict(self) -> dict:
        return {
            "answer":      self.answer,
            "steps":       [s.to_dict() for s in self.steps],
            "steps_taken": len(self.steps),
            "success":     self.success,
        }


# ---------------------------------------------------------------------------
# System prompt del agente
# ---------------------------------------------------------------------------

_SYSTEM_AGENT = """\
Eres un agente autónomo que resuelve tareas paso a paso usando herramientas.
Directorio de trabajo actual: {work_dir}

FORMATO OBLIGATORIO — sigue este esquema exactamente:

Thought: [tu razonamiento sobre qué hacer]
Action: [nombre exacto de la herramienta]
Action Input: {{"param": "valor"}}

Cuando tengas la respuesta completa escribe EXACTAMENTE:

Thought: Ya tengo toda la información necesaria.
Final Answer: [respuesta detallada y clara]

HERRAMIENTAS DISPONIBLES:
{tools}

REGLAS CRÍTICAS — JSON:
- "Action Input:" SIEMPRE debe ser un objeto JSON válido.
- CORRECTO:   Action Input: {{"command": "ls ~"}}
- INCORRECTO: Action Input: {{"raw_input": "ls ~"}}
- Usa las claves exactas que aparecen en el Ejemplo de cada herramienta.

REGLAS DE DESCUBRIMIENTO:
- Si no conoces la ruta exacta de un archivo o directorio, ejecuta primero
  list_dir con la ruta más probable (por ejemplo {{"path": "~"}}) antes de asumir.
- Usa find con -iname para búsquedas case-insensitive.
- Verifica siempre el resultado de una herramienta antes de dar Final Answer.

REGLAS GENERALES:
- Cada respuesta debe empezar con "Thought:".
- Si una herramienta falla, intenta un enfoque diferente.
- Sé breve en los pensamientos; detallado en la respuesta final.
- Máximo {max_steps} pasos en total.
- No inventes resultados. Si no sabes, dilo en Final Answer.\
"""


# ---------------------------------------------------------------------------
# Agente ReAct
# ---------------------------------------------------------------------------

class LoRAAgent:
    """
    Agente ReAct que usa cualquier función de inferencia (normalmente el
    modelo ya cargado en el servidor) para razonar, y herramientas Python
    para actuar.

    Parámetros
    ----------
    infer_fn   : función con firma _infer(messages, max_tokens, temperature, top_p) → str
    tools      : lista de Tool; si None usa DEFAULT_TOOLS
    max_steps  : número máximo de ciclos Thought→Action→Observation
    """

    # Aliases: variantes que el LLM puede generar → nombre canónico real.
    # Cubre inversiones de orden (save_note→note_save), plurales y abreviaturas
    # comunes que aparecen por contaminación de datasets externos (xLAM, ToolACE).
    _TOOL_ALIASES: Dict[str, str] = {
        # note_save
        "save_note":         "note_save",
        "note_create":       "note_save",
        "create_note":       "note_save",
        "write_note":        "note_save",
        "update_note":       "note_save",
        "notes_save":        "note_save",
        # file_organize
        "organize_files":    "file_organize",
        "files_organize":    "file_organize",
        "organize_file":     "file_organize",
        # search_files
        "file_search":       "search_files",
        "files_search":      "search_files",
        "search_file":       "search_files",
        # email_filter
        "filter_email":      "email_filter",
        "email_filters":     "email_filter",
        "emails_filter":     "email_filter",
        # calendar_get
        "get_calendar":      "calendar_get",
        "calendar_fetch":    "calendar_get",
        "fetch_calendar":    "calendar_get",
        # process_run
        "run_process":       "process_run",
        "execute_process":   "process_run",
        "process_execute":   "process_run",
        # shell (variantes comunes)
        "bash":              "shell",
        "cmd":               "shell",
        "execute":           "shell",
        "run_shell":         "shell",
        "shell_run":         "shell",
        # read_file / list_dir (variantes CamelCase)
        "ReadFile":          "read_file",
        "readFile":          "read_file",
        "ListDir":           "list_dir",
        "listDir":           "list_dir",
        "List_Dir":          "list_dir",
        # get_gpu / adapters (alucinaciones frecuentes para tareas de sistema)
        "get_gpu_memory":                   "shell",
        "get_gpu_memory_and_adapters":      "shell",
        "get_vram":                         "shell",
        "gpu_info":                         "shell",
    }

    def __init__(
        self,
        infer_fn:  Callable,
        tools:     Optional[List[Tool]] = None,
        max_steps: int = 12,
        work_dir:  str = "~",
    ) -> None:
        self._infer    = infer_fn
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or DEFAULT_TOOLS)}
        self.max_steps = max_steps
        self.work_dir  = work_dir

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_tools_desc(self) -> str:
        lines = []
        for t in self.tools.values():
            lines.append(f"- {t.name}: {t.description}\n  Ejemplo: {t.params_doc}")
        return "\n".join(lines)

    def _build_system(self) -> str:
        return _SYSTEM_AGENT.format(
            tools     = self._format_tools_desc(),
            max_steps = self.max_steps,
            work_dir  = self.work_dir,
        )

    def _parse_output(
        self, text: str
    ) -> Tuple[str, Optional[str], Optional[dict], bool, Optional[str]]:
        """
        Parsea la respuesta del LLM.
        Devuelve: (thought, action, action_input, is_final, final_answer)
        """
        thought      = ""
        action       = None
        action_input = None
        is_final     = False
        final_answer = None

        # Thought (puede ser vacío)
        m = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if m:
            thought = m.group(1).strip()

        # Final Answer — tiene prioridad
        m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return thought, None, None, True, m.group(1).strip()

        # Action
        m = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        if m:
            action = m.group(1).strip()

        # Action Input — extraer objeto JSON con balance de llaves
        # (el regex simple {.+?} falla cuando el valor contiene llaves,
        # por ejemplo comandos con awk '{print $2}')
        ai_match = re.search(r"Action Input:\s*\{", text, re.IGNORECASE)
        if ai_match:
            raw = _extract_json_object(text, ai_match.end() - 1)
            if raw:
                try:
                    action_input = json.loads(raw)
                except json.JSONDecodeError:
                    # Reparar comillas simples → dobles como último recurso
                    try:
                        action_input = json.loads(raw.replace("'", '"'))
                    except json.JSONDecodeError:
                        action_input = {"raw_input": raw}

        # Rescate raw_input: el modelo generó {"raw_input": "..."} en vez del
        # JSON correcto. Intentar parsear el valor como JSON real.
        if (
            action_input is not None
            and list(action_input.keys()) == ["raw_input"]
        ):
            raw_val = action_input["raw_input"]
            try:
                rescued = json.loads(raw_val)
                if isinstance(rescued, dict):
                    action_input = rescued
            except (json.JSONDecodeError, TypeError):
                # No es JSON — el valor puede ser un comando de texto plano.
                # Para herramientas como `shell` esto es válido como {"command": raw_val}.
                if action and action in {"shell", "bash", "cmd"}:
                    action_input = {"command": raw_val}

        return thought, action, action_input, is_final, final_answer

    # ------------------------------------------------------------------
    # Público
    # ------------------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """
        Ejecuta el agente hasta Final Answer o max_steps.

        Parámetros
        ----------
        task : descripción de la tarea en lenguaje natural

        Devuelve
        --------
        AgentResult con la respuesta final y la traza de pasos

        Estrategia de inferencia en dos fases
        --------------------------------------
        Fase 1 — texto libre
            El modelo razona (Thought) y elige la herramienta (Action).
            Sin restricciones de gramática → el modelo puede "pensar" libremente.

        Fase 2 — JSON Schema constrained (solo si la herramienta tiene input_schema)
            Se añade al historial la salida de Fase 1 y se fuerza al modelo a
            generar **únicamente** el JSON del Action Input, conforme al schema.
            llama-cpp-python garantiza matemáticamente tokens válidos.
            Si la función de inferencia no soporta grammar_schema (HuggingFace o
            versión vieja de llama-cpp), se usa el parser JSON de 5 niveles
            como fallback (comportamiento heredado).

        Truncado de Observations antiguas
        ----------------------------------
        Para proteger la ventana de contexto en bucles largos (≥ 3 pasos), las
        observations de más de 2 iteraciones atrás se acortan a ≤ 500 caracteres.
        Las 2 iteraciones más recientes (Thought + Observation) se mantienen
        íntegras para que el modelo recuerde lo que acaba de hacer.
        """
        system   = self._build_system()
        history: List[dict] = [{"role": "user", "content": f"Tarea: {task}"}]
        steps:   List[AgentStep] = []

        for step_idx in range(self.max_steps):

            # ── Truncado adaptativo de Observations antiguas ──────────────
            # El historial alterna: user (Tarea/Observation) y assistant (Thought+Action).
            # Comprimimos las Observations más antiguas para proteger el contexto.
            #   • Guardamos intactas las 2 últimas iteraciones (4 últimos mensajes
            #     tras el mensaje de tarea inicial: assistant+user+assistant+user).
            #   • El resto de mensajes "Observation:" del user los truncamos a 500 chars.
            if step_idx >= 2 and len(history) > 5:
                # Los primeros N-4 mensajes (excluida la tarea inicial) son "antiguos"
                for i, msg in enumerate(history[:-4]):  # no tocar los 4 más recientes
                    if i == 0:
                        continue  # el mensaje de tarea inicial: nunca truncar
                    if msg["role"] == "user" and msg["content"].startswith("Observation:"):
                        obs_body = msg["content"][len("Observation:"):]
                        if len(obs_body) > 500:
                            history[i] = {
                                "role": "user",
                                "content": "Observation:" + obs_body[:500] + "\n[...truncado en contexto antiguo]",
                            }
            # ──────────────────────────────────────────────────────────────

            messages = [{"role": "system", "content": system}] + history

            # ── Fase 1: inferencia de texto libre ─────────────────────────
            try:
                llm_out = self._infer(
                    messages,
                    max_tokens  = 1024,
                    temperature = 0.0,   # greedy: 100 % reproducible para benchmark
                    top_p       = 1.0,
                )
            except Exception as e:
                return AgentResult(
                    answer  = f"Error en inferencia: {e}",
                    steps   = steps,
                    success = False,
                )

            history.append({"role": "assistant", "content": llm_out})
            thought, action, action_input, is_final, final_answer = self._parse_output(llm_out)

            # ── Final Answer ──────────────────────────────────────────
            if is_final:
                motivo = _sanitize_output(final_answer)
                if motivo:
                    steps.append(AgentStep(thought=thought, is_final=True, final_answer=final_answer))
                    return AgentResult(
                        answer  = f"[Error de sanitización en Final Answer]: {motivo}\nRespuesta generada: {final_answer}",
                        steps   = steps,
                        success = False,
                    )
                steps.append(AgentStep(thought=thought, is_final=True, final_answer=final_answer))
                return AgentResult(answer=final_answer or "", steps=steps, success=True)

            # ── Sin action: el modelo no siguió el formato ────────────
            if not action:
                hint = (
                    "Observation: [Error de formato] "
                    "Debes responder con:\n"
                    "  Thought: ...\n"
                    "  Action: <nombre_herramienta>\n"
                    "  Action Input: {\"param\": \"valor\"}\n"
                    "O si ya tienes la respuesta:\n"
                    "  Thought: ...\n"
                    "  Final Answer: <respuesta>"
                )
                steps.append(AgentStep(
                    thought     = thought or llm_out[:150],
                    observation = hint,
                ))
                history.append({"role": "user", "content": hint})
                continue

            # ── Fase 2: forzar JSON correcto si la herramienta tiene schema ─
            # Comprobamos si la acción elegida tiene input_schema.
            resolved_action = self._TOOL_ALIASES.get(action, action)
            tool_obj = self.tools.get(resolved_action)

            if tool_obj is not None and getattr(tool_obj, "input_schema", None) is not None and action_input is None:
                # El parser no extrajo JSON válido: usar gramática para forzarlo.
                # Construimos un prompt de continuación que pide SOLO el JSON del input.
                try:
                    phase2_messages = messages + [
                        {"role": "assistant", "content": llm_out},
                        {"role": "user",
                         "content": (
                             "Provide the Action Input JSON for the tool call above. "
                             "Respond with ONLY valid JSON, no explanation."
                         )},
                    ]
                    raw_json = self._infer(
                        phase2_messages,
                        max_tokens     = 256,
                        temperature    = 0.0,
                        top_p          = 1.0,
                        grammar_schema = tool_obj.input_schema,
                    )
                    import json as _json
                    action_input = _json.loads(raw_json.strip())
                except Exception:
                    action_input = {}   # no se pudo obtener JSON; la herramienta lo manejará

            # ── Ejecutar herramienta ──────────────────────────────────
            if resolved_action in self.tools:
                observation = self.tools[resolved_action](**(action_input or {}))
            else:
                available = list(self.tools.keys())
                observation = (
                    f"[Herramienta desconocida]: '{action}'. "
                    f"Disponibles: {available}"
                )

            # Sanitización de observación (versión laxa: solo vacío o {{...}})
            motivo_obs = _sanitize_observation(observation)
            if motivo_obs:
                steps.append(AgentStep(
                    thought      = thought,
                    action       = resolved_action,
                    action_input = action_input,
                    observation  = observation,
                ))
                return AgentResult(
                    answer  = f"[Error de sanitización en Observation]: {motivo_obs}\nObservación generada: {observation}",
                    steps   = steps,
                    success = False,
                )

            # Truncar observaciones muy largas
            if len(observation) > 2_500:
                observation = observation[:2_500] + "\n[...truncado]"

            steps.append(AgentStep(
                thought      = thought,
                action       = resolved_action,   # nombre canónico real
                action_input = action_input,
                observation  = observation,
            ))
            history.append({"role": "user", "content": f"Observation: {observation}"})

        # Pasos agotados
        return AgentResult(
            answer  = "Se agotaron los pasos sin llegar a una respuesta final.",
            steps   = steps,
            success = False,
        )
