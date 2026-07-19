"""
motor.log_quality
=================
Filtro de calidad para el interaction_log.jsonl antes de usarlo como dataset
de entrenamiento (learn --auto) o de pares de preferencia (DPOBuilder).

Por qué existe (datos reales, 12-jun-2026): el log de uso vía Odysseus contiene
ruido sistemático que envenenaría un adapter si entrara sin filtrar:
  - Respuestas vacías o basura: "", "[]", "null", "{}", "```json\\n[]\\n```"
    (el doble-envío de Odysseus deja una respuesta buena y una basura).
  - Respuestas truncadas (finish_reason="length") — entrenar con texto cortado
    a media palabra enseña al modelo a no terminar las frases.
  - Tool calls (assistant vacío + finish_reason="tool_calls") — no son ejemplos
    de texto SFT; necesitarían un pipeline de tool-calling aparte.
  - Mensajes de usuario duplicados (reintentos del cliente).

Diseño: una sola función de veredicto (`quality_check`) con motivo explícito,
y dos cargadores (`load_sft_examples`, `load_quality_entries`) que ambos
caminos de ingesta comparten. El veredicto nunca lanza; un log corrupto no
debe romper el entrenamiento, solo descartar la línea mala.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Respuestas que son "no-respuesta" una vez quitado el formato. En minúsculas.
_JUNK_TOKENS = {"", "[]", "{}", "null", "none", "n/a", "[done]", "...", "."}

# Longitud mínima de una respuesta útil (caracteres, tras limpiar). Por debajo
# casi siempre es ruido ("ok", "[]", un emoji suelto). 15 sale de la inspección
# del log real: separa basura de respuestas legítimas más cortas observadas.
DEFAULT_MIN_CHARS = 15

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_]*\s*|\s*```$")


def clean_text(text: Optional[str]) -> str:
    """Normaliza una respuesta: quita fences de código markdown y espacios."""
    if not text:
        return ""
    t = str(text).strip()
    # Quitar un único bloque ```lang ... ``` que envuelva todo el contenido
    if t.startswith("```") and t.endswith("```"):
        t = _FENCE_RE.sub("", t)
        t = _FENCE_RE.sub("", t).strip()
    return t.strip()


def is_junk(text: Optional[str]) -> bool:
    """True si la respuesta es vacía o un 'no-contenido' (tras limpiar)."""
    return clean_text(text).lower() in _JUNK_TOKENS


def quality_check(
    entry: dict,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    reject_truncated: bool = True,
) -> Tuple[bool, str]:
    """
    Decide si una entrada del log sirve como ejemplo de entrenamiento de texto.

    Devuelve (ok, motivo). Si ok es False, `motivo` explica por qué se descarta
    (útil para auditar el filtrado). Nunca lanza.
    """
    if not isinstance(entry, dict):
        return False, "no_dict"

    # str() defensivo: un log real acumula user_msg no-string (int, lista) por
    # clientes que mandan basura; sin esto .strip() lanzaba AttributeError y
    # tumbaba TODO el filtrado (y con él learn --auto / DPO). "Nunca lanza" es
    # el contrato — una entrada mal tipada se descarta, no rompe el lote.
    user = str(entry.get("user_msg") or "").strip()
    assistant_raw = entry.get("assistant")

    if not user:
        return False, "user_vacio"

    # Tool calls: assistant vacío + tool_calls presentes → no es SFT de texto
    if entry.get("tool_calls") or entry.get("finish_reason") == "tool_calls":
        return False, "tool_call"

    if reject_truncated and entry.get("finish_reason") == "length":
        return False, "truncado"

    if is_junk(assistant_raw):
        return False, "assistant_basura"

    assistant = clean_text(assistant_raw)
    if len(assistant) < min_chars:
        return False, "assistant_corto"

    return True, "ok"


def load_quality_entries(
    log_path,
    *,
    feedback: Optional[set] = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    reject_truncated: bool = True,
    dedup: bool = True,
    report: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """
    Carga las entradas del log que pasan el filtro de calidad.

    feedback : si se indica (p.ej. {1} o {1, -1}), solo entradas cuyo
               campo feedback esté en ese conjunto. None = sin filtrar por
               feedback (learn --auto acepta None y 1 — ver wrapper abajo).
    dedup    : descarta pares (user_msg, assistant) idénticos repetidos.
    report   : si se pasa un dict, se rellena con el conteo por motivo de
               descarte (para imprimir un resumen del filtrado).
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"[log_quality] Log no encontrado: {path}")

    counts: Dict[str, int] = report if report is not None else {}

    def _bump(reason: str) -> None:
        counts[reason] = counts.get(reason, 0) + 1

    seen: set = set()
    kept: List[dict] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                _bump("json_invalido")
                continue

            if feedback is not None and entry.get("feedback") not in feedback:
                _bump("feedback_no_coincide")
                continue

            ok, reason = quality_check(
                entry, min_chars=min_chars, reject_truncated=reject_truncated
            )
            if not ok:
                _bump(reason)
                continue

            if dedup:
                key = (
                    (entry.get("user_msg") or "").strip().lower(),
                    clean_text(entry.get("assistant")),
                )
                if key in seen:
                    _bump("duplicado")
                    continue
                seen.add(key)

            _bump("ok")
            kept.append(entry)

    return kept


def load_sft_examples(
    log_path,
    *,
    include_unrated: bool = True,
    min_chars: int = DEFAULT_MIN_CHARS,
    report: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """
    Ejemplos limpios en formato chat para SFT (learn --auto).

    include_unrated=True: acepta feedback None (implícito aceptado) y 1, como
    hacía learn --auto, pero ahora pasando el filtro de calidad. feedback=-1
    (pulgar abajo) SIEMPRE se excluye.

    Devuelve [{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}]
    """
    fb = {None, 1} if include_unrated else {1}
    entries = load_quality_entries(
        log_path, feedback=fb, min_chars=min_chars, report=report
    )
    return [
        {
            "messages": [
                {"role": "user",      "content": (e.get("user_msg") or "").strip()},
                {"role": "assistant", "content": clean_text(e.get("assistant"))},
            ]
        }
        for e in entries
    ]


def format_report(report: Dict[str, int]) -> str:
    """Resumen legible del filtrado para imprimir tras cargar."""
    total = sum(report.values())
    ok = report.get("ok", 0)
    lines = [f"[log_quality] {ok}/{total} entradas pasaron el filtro de calidad"]
    descartes = {k: v for k, v in report.items() if k != "ok" and v}
    if descartes:
        detalle = ", ".join(f"{k}={v}" for k, v in sorted(descartes.items()))
        lines.append(f"[log_quality]   descartadas: {detalle}")
    return "\n".join(lines)
