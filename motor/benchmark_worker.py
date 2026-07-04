"""
motor.benchmark_worker
======================
Valida un GGUF candidato con un conjunto de tareas estándar antes de promoverlo.

Las tareas testean las capacidades core del agente doméstico:
  1. note_save       — guardar una nota (herramienta básica, 1 paso)
  2. file_organize   — organizar archivos (herramienta básica, 1 paso)
  3. search_files    — búsqueda de archivos (herramienta básica, 1 paso)
  4. multi_step      — tarea encadenada: buscar → leer → guardar (2+ pasos)
  5. json_format     — el agente responde JSON válido (robustez de formato)

Umbral de aprobación: ≥ 4 de 5 tareas correctas (80%).

Uso
---
  from motor.benchmark_worker import run_benchmark

  passed, report = run_benchmark("modelos/candidate.gguf")
  # passed: bool
  # report: {"note_save": True, "file_organize": False, ...}
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Umbral
# ─────────────────────────────────────────────────────────────────────────────
PASS_THRESHOLD = 0.80   # >= 80% de tareas correctas


# ─────────────────────────────────────────────────────────────────────────────
# Definición de tareas
# ─────────────────────────────────────────────────────────────────────────────

TASKS = {
    "note_save": {
        "prompt": (
            "Guarda una nota con el texto: 'Reunión mañana a las 10h con el equipo'."
        ),
        # Esperamos que el agente llame a note_save con el texto correcto
        "check": lambda resp: _tool_called(resp, "note_save") and "reunión" in resp.lower(),
    },
    "file_organize": {
        "prompt": (
            "Organiza los archivos de la carpeta ~/Descargas moviendo los PDF a ~/Documentos/PDFs/."
        ),
        "check": lambda resp: _tool_called(resp, "file_organize") or _tool_called(resp, "move_file"),
    },
    "search_files": {
        "prompt": "Busca todos los archivos .txt en ~/Documentos.",
        "check": lambda resp: _tool_called(resp, "search_files") and ".txt" in resp,
    },
    "multi_step": {
        "prompt": (
            "Lista los archivos en ~/Notas, luego guarda una nota con el nombre del primer archivo que encuentres."
        ),
        # Debe llamar list_dir/search_files Y note_save
        "check": lambda resp: (
            (_tool_called(resp, "list_dir") or _tool_called(resp, "search_files"))
            and _tool_called(resp, "note_save")
        ),
    },
    "json_format": {
        "prompt": "¿Qué herramientas tienes disponibles? Responde en formato JSON.",
        # Solo validamos que la respuesta contenga JSON parseable
        "check": lambda resp: _contains_valid_json(resp),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de validación
# ─────────────────────────────────────────────────────────────────────────────

def _tool_called(response: str, tool_name: str) -> bool:
    """Comprueba si el nombre de herramienta aparece en la respuesta del agente."""
    return tool_name.lower() in response.lower()


def _contains_valid_json(text: str) -> bool:
    """Devuelve True si el texto contiene al menos un bloque JSON válido."""
    # Busca bloques ```json ... ``` o JSON inline { ... }
    candidates = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    candidates += re.findall(r"(\{[^{}]{10,}\})", text, re.DOTALL)
    for c in candidates:
        try:
            json.loads(c)
            return True
        except (json.JSONDecodeError, ValueError):
            continue
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia con llama-cpp-python
# ─────────────────────────────────────────────────────────────────────────────

def _load_gguf(model_path: str):
    """Carga el GGUF en CPU para inferencia de benchmark."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "[Benchmark] llama-cpp-python no instalado.\n"
            "Ejecuta: pip install llama-cpp-python"
        )
    import os
    return Llama(
        model_path = model_path,
        n_ctx      = 2048,
        n_threads  = os.cpu_count() or 4,
        verbose    = False,
    )


def _infer(llm, prompt: str, max_tokens: int = 512) -> str:
    """Genera una respuesta dado un prompt."""
    result = llm.create_chat_completion(
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = max_tokens,
        temperature = 0.1,   # baja temperatura para mayor determinismo en el test
    )
    return result["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint público
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(model_path: str) -> Tuple[bool, Dict[str, bool]]:
    """
    Ejecuta las 5 tareas estándar contra el GGUF en `model_path`.

    Returns
    -------
    passed : bool
        True si el candidato supera el umbral (≥ 80% tareas correctas).
    report : dict[str, bool]
        Resultado individual por tarea.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"[Benchmark] No se encuentra el modelo: {model_path}")

    print(f"[Benchmark] Cargando modelo: {model_path}")
    t0  = time.time()
    llm = _load_gguf(model_path)
    print(f"[Benchmark] Modelo cargado en {time.time() - t0:.1f}s")

    report: Dict[str, bool] = {}
    for task_name, task in TASKS.items():
        print(f"[Benchmark] Tarea: {task_name} ...", end=" ", flush=True)
        try:
            response = _infer(llm, task["prompt"])
            ok       = task["check"](response)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR ({exc})")
            ok = False
        report[task_name] = ok
        print("✅" if ok else "❌")

    n_passed  = sum(report.values())
    n_total   = len(report)
    ratio     = n_passed / n_total
    passed    = ratio >= PASS_THRESHOLD

    print(
        f"\n[Benchmark] {n_passed}/{n_total} tareas correctas "
        f"({ratio:.0%}) — {'APROBADO ✅' if passed else 'SUSPENDIDO ❌'}"
    )
    return passed, report


# ─────────────────────────────────────────────────────────────────────────────
# CLI standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark del GGUF candidato")
    parser.add_argument("model_path", help="Ruta al archivo .gguf a evaluar")
    args = parser.parse_args()

    passed, report = run_benchmark(args.model_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)
