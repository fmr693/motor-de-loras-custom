"""
motor.reflection
================
Pase de reflexión: feedback IMPLÍCITO por LLM-juez (aprendizaje híbrido).

Inspirado en el `background_review` de Hermes Agent: en vez de depender solo
del feedback explícito (👍/👎, que es escaso porque casi nadie pulsa), un LLM
relee las conversaciones del `interaction_log.jsonl` y deduce de las señales
naturales qué respuestas fueron acierto o error:

  • El usuario reformuló la misma pregunta         → la respuesta anterior falló
  • El usuario corrigió / se frustró / dijo "no"   → error (señal de 1ª clase)
  • El usuario dio las gracias / siguió adelante /
    construyó sobre la respuesta                    → acierto
  • Sin señal clara                                 → neutral (se ignora)

Salidas (complementan, no reemplazan, al feedback explícito de `/feedback`):
  1. Etiquetas inferidas por interacción → `reflection_labels.jsonl`
     {"id", "session_id", "turn", "label": +1|-1, "confidence", "reason",
      "source": "reflection"}
  2. Pares de preferencia de correcciones → se fusionan con DPOBuilder
     {"prompt", "chosen", "rejected", "source": "reflection"}

El modelo juez es configurable (como en Hermes): por defecto la Gemma local
(soberano, gratis), pero se puede apuntar a un modelo más fuerte para juzgar
mejor — un 12B juzgándose a sí mismo tiene sesgo, así que la confianza baja
se trata con cautela.

  MOTOR_JUDGE_URL    endpoint OpenAI del juez (def http://localhost:8001/v1)
  MOTOR_JUDGE_MODEL  id del modelo juez (def gemma-4-12B-it-Q4_K_M.gguf)

CLI (previsto en fabrica_loras.py):
  python fabrica_loras.py reflect --log logs/interaction_log.jsonl \\
         --out datasets/reflection --min-confidence 0.6
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Cliente LLM mínimo (OpenAI-compatible, sin dependencias)
# ---------------------------------------------------------------------------

def _judge_url() -> str:
    return os.environ.get("MOTOR_JUDGE_URL", "http://localhost:8001/v1").rstrip("/")


def _judge_model() -> str:
    return os.environ.get("MOTOR_JUDGE_MODEL", "gemma-4-12B-it-Q4_K_M.gguf")


def _chat(messages: List[dict], *, max_tokens: int = 1200,
          temperature: float = 0.0, timeout: int = 180) -> str:
    """Llamada de chat OpenAI-compatible. Devuelve el texto de la respuesta."""
    body = json.dumps({
        "model": _judge_model(), "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        _judge_url() + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"] or ""


# ---------------------------------------------------------------------------
# Prompt del juez (el corazón — análogo al _SKILL_REVIEW_PROMPT de Hermes)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "Eres un evaluador de calidad de conversaciones. Tu trabajo es leer una "
    "conversación entre un usuario y un asistente de IA y deducir, SOLO a "
    "partir de señales implícitas del propio usuario, si cada respuesta del "
    "asistente fue un ACIERTO o un ERROR. No juzgas la respuesta por tu "
    "propio criterio: juzgas cómo reaccionó el usuario a ella.\n\n"
    "Señales de ERROR (la respuesta falló):\n"
    "  - El usuario reformula o repite la misma pregunta.\n"
    "  - El usuario corrige al asistente ('no', 'te equivocas', 'en realidad…').\n"
    "  - El usuario muestra frustración ('otra vez no', 'eso no es lo que pedí').\n"
    "  - El usuario señala un fallo concreto o pide rehacerlo.\n\n"
    "Señales de ACIERTO (la respuesta funcionó):\n"
    "  - El usuario da las gracias o expresa satisfacción.\n"
    "  - El usuario avanza al siguiente tema o construye sobre la respuesta.\n"
    "  - El usuario confirma ('perfecto', 'eso es', 'funcionó').\n\n"
    "Si no hay ninguna señal clara, la etiqueta es 'neutral' (no inventes).\n"
    "Cuando el usuario CORRIGE una respuesta y en su turno queda claro cuál "
    "era la respuesta correcta, extrae un par de preferencia.\n\n"
    "Responde EXCLUSIVAMENTE con un objeto JSON válido, sin texto alrededor, "
    "con esta forma:\n"
    "{\n"
    '  "verdicts": [\n'
    '    {"turn": <int>, "label": "acierto"|"error"|"neutral", '
    '"confidence": <0.0-1.0>, "reason": "<breve, en qué señal te basas>"}\n'
    "  ],\n"
    '  "correction_pairs": [\n'
    '    {"turn": <int>, "rejected": "<respuesta original del asistente, '
    'resumida si es larga>", "chosen": "<respuesta correcta implícita en la '
    'corrección del usuario>"}\n'
    "  ]\n"
    "}"
)


# ---------------------------------------------------------------------------
# Estructuras y agrupación
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    id: str
    session_id: str
    turn: int
    label: int           # +1 acierto, -1 error
    confidence: float
    reason: str
    source: str = "reflection"


@dataclass
class ReflectionResult:
    labels: List[Verdict] = field(default_factory=list)
    pairs: List[Dict[str, str]] = field(default_factory=list)
    sessions_judged: int = 0
    sessions_failed: int = 0

    def counts(self) -> Dict[str, int]:
        pos = sum(1 for v in self.labels if v.label == 1)
        neg = sum(1 for v in self.labels if v.label == -1)
        return {"acierto": pos, "error": neg, "pares": len(self.pairs)}


def _load_sessions(log_path: Path) -> Dict[str, List[dict]]:
    """Agrupa las entradas del log en conversaciones, ordenadas por turno.

    - Entradas con `session_id` (endpoint /chat/session) → conversación
      multi-turno real: el juez puede leer la reacción del usuario.
    - Entradas sin `session_id` (endpoint /v1, Odysseus/Hermes) → NO tienen
      enlace conversacional en el log, así que cada una es su propia sesión
      de un turno (clavada por su id único). NO se lumpan en una conversación
      ficticia. Enlazarlas requiere un fingerprint de conversación en el
      logging de /v1 (pendiente — es lo que desbloquea la señal en el uso real).

    Ignora entradas sin user_msg/assistant (ruido de doble-envío)."""
    sessions: Dict[str, List[dict]] = defaultdict(list)
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue  # línea corrupta → se ignora, no es fatal
            if not (e.get("user_msg") and e.get("assistant")):
                continue
            sid = e.get("session_id")
            key = sid if sid else f"solo:{e.get('id')}"
            sessions[key].append(e)
    for key in sessions:
        sessions[key].sort(key=lambda x: x.get("turn") or 0)
    return sessions


def _render_conversation(turns: List[dict], max_chars: int = 1200) -> str:
    """Renderiza una sesión como texto legible para el juez."""
    out = []
    for e in turns:
        u = (e.get("user_msg") or "").strip()
        a = (e.get("assistant") or "").strip()
        if len(a) > max_chars:
            a = a[:max_chars] + " […]"
        if len(u) > max_chars:
            u = u[:max_chars] + " […]"
        out.append(f"[turno {e.get('turn', 0)}]\nUSUARIO: {u}\nASISTENTE: {a}")
    return "\n\n".join(out)


def _extract_json(text: str) -> Optional[dict]:
    """Extrae el primer objeto JSON del texto del juez (tolerante a envoltura)."""
    text = text.strip()
    # Quitar fences de markdown si los hubiera
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Buscar el primer bloque {...} equilibrado
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


_LABEL_MAP = {"acierto": 1, "error": -1}

# Troceado de sesiones largas: una sesión agéntica de decenas de turnos genera
# un JSON de veredictos que desborda max_tokens y se trunca (→ parse fail). Se
# juzga por ventanas solapadas para que la reacción del usuario a una respuesta
# (que vive en el turno siguiente) nunca quede partida por el corte.
_CHUNK_TURNS = 10
_CHUNK_OVERLAP = 1


def _chunk(turns: List[dict], size: int, overlap: int) -> List[List[dict]]:
    """Divide una lista de turnos en ventanas solapadas de tamaño `size`."""
    if len(turns) <= size:
        return [turns]
    step = max(1, size - overlap)
    out: List[List[dict]] = []
    i = 0
    while i < len(turns):
        out.append(turns[i:i + size])
        if i + size >= len(turns):
            break
        i += step
    return out


# ---------------------------------------------------------------------------
# Juez de reflexión
# ---------------------------------------------------------------------------

class ReflectionJudge:
    """Relee el interaction_log y deduce feedback implícito con un LLM-juez."""

    def __init__(self, log_path: str | Path, min_confidence: float = 0.6):
        self.log_path = Path(log_path)
        self.min_confidence = min_confidence

    def run(self) -> ReflectionResult:
        if not self.log_path.exists():
            raise FileNotFoundError(f"Log no encontrado: {self.log_path}")
        sessions = _load_sessions(self.log_path)
        result = ReflectionResult()

        for sid, turns in sessions.items():
            # La señal implícita vive en la REACCIÓN del usuario al turno
            # siguiente. Una interacción suelta (1 turno) no la tiene → se
            # salta (además evita gastar una llamada al juez por cada /v1
            # histórico sin enlazar).
            if len(turns) < 2:
                continue
            by_turn = {(e.get("turn") or 0): e for e in turns}
            # Dedup de veredictos por turno (las ventanas solapan): se queda el
            # de mayor confianza.
            best: Dict[int, Verdict] = {}
            seen_pairs: set = set()
            session_ok = False

            for window in _chunk(turns, _CHUNK_TURNS, _CHUNK_OVERLAP):
                convo = _render_conversation(window)
                try:
                    raw = _chat([
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user", "content":
                            "Conversación a evaluar:\n\n" + convo +
                            "\n\nDevuelve el JSON de veredictos."},
                    ])
                except Exception:
                    continue
                parsed = _extract_json(raw)
                if not parsed:
                    continue
                session_ok = True

                for v in parsed.get("verdicts", []):
                    label = _LABEL_MAP.get(str(v.get("label", "")).lower())
                    conf = float(v.get("confidence", 0) or 0)
                    if label is None or conf < self.min_confidence:
                        continue
                    turn = int(v.get("turn", 0) or 0)
                    prev = best.get(turn)
                    if prev is not None and prev.confidence >= conf:
                        continue
                    entry = by_turn.get(turn, {})
                    best[turn] = Verdict(
                        id=entry.get("id", f"{sid}_t{turn}"),
                        session_id=sid, turn=turn, label=label,
                        confidence=conf, reason=str(v.get("reason", "")),
                    )

                for p in parsed.get("correction_pairs", []):
                    turn = int(p.get("turn", 0) or 0)
                    entry = by_turn.get(turn, {})
                    prompt = entry.get("user_msg", "")
                    chosen = str(p.get("chosen", "")).strip()
                    rejected = str(p.get("rejected", "")).strip()
                    key = (prompt, rejected)
                    if (prompt and chosen and rejected and chosen != rejected
                            and key not in seen_pairs):
                        seen_pairs.add(key)
                        result.pairs.append({
                            "prompt": prompt, "chosen": chosen,
                            "rejected": rejected, "source": "reflection",
                        })

            if session_ok:
                result.sessions_judged += 1
                result.labels.extend(best.values())
            else:
                result.sessions_failed += 1

        return result

    def write(self, result: ReflectionResult, out_dir: str | Path) -> Dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        labels_path = out / "reflection_labels.jsonl"
        pairs_path = out / "reflection_pairs.jsonl"
        with open(labels_path, "w", encoding="utf-8") as fh:
            for v in result.labels:
                fh.write(json.dumps(v.__dict__, ensure_ascii=False) + "\n")
        with open(pairs_path, "w", encoding="utf-8") as fh:
            for p in result.pairs:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        return {"labels": labels_path, "pairs": pairs_path}


def format_report(result: ReflectionResult) -> str:
    c = result.counts()
    return (
        f"Reflexión: {result.sessions_judged} sesiones juzgadas "
        f"({result.sessions_failed} sin veredicto) → "
        f"{c['acierto']} aciertos, {c['error']} errores inferidos, "
        f"{c['pares']} pares de corrección."
    )
