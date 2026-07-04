"""
motor.domestic_dataset_gen
==========================
Generador del dataset base para el agente doméstico (Sprint 8).

Arquitectura: representación semántica interna model-agnostic + renderizadores
por familia de modelo. Los ejemplos se almacenan como dicts estructurados y se
convierten al formato del modelo objetivo al exportar.

Renderers disponibles
---------------------
  to_react(ex)              → str   ReAct Thought/Action/Observation (universal)
  to_qwen_native(ex)        → list  ChatML con <tool_call>/<tool_response>
  to_chatml(ex, renderer)   → dict  Ejemplo completo para SFT {messages: [...]}

Uso rápido
----------
  from motor.domestic_dataset_gen import generate_dataset
  generate_dataset(n=2000, target_model="multiturn", out="dataset_domestic.jsonl")

  # Adaptar ejemplos de xlam:
  from motor.domestic_dataset_gen import adapt_xlam
  extras = adapt_xlam("xlam_60k.jsonl", n=600)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional, TypedDict


# ---------------------------------------------------------------------------
# Tipo canónico de ejemplo (representación semántica interna)
# ---------------------------------------------------------------------------

class Step(TypedDict):
    thought:  str   # razonamiento del modelo antes de llamar la herramienta
    tool:     str   # nombre exacto de la herramienta
    args:     dict  # parámetros de la llamada (los que irán en Action Input)
    result:   str   # observación devuelta por la herramienta


class DomesticExample(TypedDict):
    user_request:  str         # petición del usuario en lenguaje natural
    steps:         list[Step]  # uno o más pasos tool-call
    final_thought: str         # último razonamiento antes de Final Answer
    final_answer:  str         # respuesta final al usuario
    language:      str         # "es" | "en"
    category:      str         # herramienta principal: file_organize, email_filter, …


# ---------------------------------------------------------------------------
# System prompt del agente doméstico
# (reusa el template de motor.agent._SYSTEM_AGENT con las domestic tools)
# ---------------------------------------------------------------------------

_DOMESTIC_SYSTEM: str | None = None  # cache — se inicializa en primer uso


def _build_domestic_system() -> str:
    global _DOMESTIC_SYSTEM
    if _DOMESTIC_SYSTEM is None:
        from motor.agent import _SYSTEM_AGENT
        from motor.domestic_tools import DOMESTIC_TOOLS
        tools_desc = "\n".join(
            f"- {t.name}: {t.description}\n  Ejemplo: {t.params_doc}"
            for t in DOMESTIC_TOOLS
        )
        _DOMESTIC_SYSTEM = _SYSTEM_AGENT.format(tools=tools_desc, max_steps=10, work_dir="~")
    return _DOMESTIC_SYSTEM


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def to_react(ex: DomesticExample) -> str:
    """
    Convierte un DomesticExample al texto completo del turno assistant en ReAct.
    Formato compatible con motor.agent._parse_output().

    Thought: ...
    Action: tool_name
    Action Input: {"key": "value"}
    Observation: ...
    ...
    Thought: final_thought
    Final Answer: final_answer
    """
    parts: list[str] = []
    for step in ex["steps"]:
        parts.append(f"Thought: {step['thought']}")
        parts.append(f"Action: {step['tool']}")
        parts.append(f"Action Input: {json.dumps(step['args'], ensure_ascii=False)}")
        parts.append(f"Observation: {step['result']}")
    parts.append(f"Thought: {ex['final_thought']}")
    parts.append(f"Final Answer: {ex['final_answer']}")
    return "\n".join(parts)


def to_qwen_native(ex: DomesticExample) -> list[dict]:
    """
    Convierte un DomesticExample a la lista de mensajes ChatML con el formato
    nativo de tool-calling de Qwen (segunda iteración, requiere reescribir el
    parser de motor.agent).

    Devuelve una lista de dicts con role/content para el turno assistant + tool.
    """
    messages: list[dict] = []
    for step in ex["steps"]:
        # turno assistant: razonamiento + tool_call
        tool_call_block = (
            f"<tool_call>\n"
            f'{{"name": "{step["tool"]}", "arguments": '
            f'{json.dumps(step["args"], ensure_ascii=False)}}}\n'
            f"</tool_call>"
        )
        messages.append({
            "role": "assistant",
            "content": f"{step['thought']}\n{tool_call_block}",
        })
        # turno tool: resultado
        messages.append({
            "role": "tool",
            "name": step["tool"],
            "content": step["result"],
        })

    # último turno assistant: respuesta final
    messages.append({
        "role": "assistant",
        "content": ex["final_answer"],
    })
    return messages


def to_chatml(
    ex:       DomesticExample,
    renderer: str = "react",
) -> dict:
    """
    Envuelve un DomesticExample en el formato ChatML completo para SFT:
    {"messages": [system, user, assistant, ...]}.

    renderer: "react" | "qwen_native" | "multiturn"
    """
    system = _build_domestic_system()

    if renderer == "react":
        return {
            "messages": [
                {"role": "system",    "content": system},
                {"role": "user",      "content": f"Tarea: {ex['user_request']}"},
                {"role": "assistant", "content": to_react(ex)},
            ]
        }

    if renderer == "qwen_native":
        from motor.domestic_tools import DOMESTIC_TOOLS
        tools_schema = [
            {"type": "function", "function": {"name": t.name, "description": t.description}}
            for t in DOMESTIC_TOOLS
        ]
        base = [
            {"role": "system", "content": system},
            {"role": "user",   "content": ex["user_request"]},
        ]
        return {"messages": base + to_qwen_native(ex), "tools": tools_schema}

    if renderer == "multiturn":
        return to_multiturn(ex)

    raise ValueError(f"Renderer desconocido: '{renderer}'. Usa 'react', 'qwen_native' o 'multiturn'.")


def to_multiturn(ex: DomesticExample) -> dict:
    """
    Convierte un DomesticExample al formato ChatML multi-turno real, que refleja
    cómo funciona el agente en inferencia:

      system
      user:      "Tarea: ..."
      assistant: "Thought: ...\nAction: ...\nAction Input: {...}"   ← genera hasta aquí
      user:      "Observation: ..."                                  ← env. devuelve resultado
      assistant: "Thought: ...\nAction: ...\nAction Input: {...}"   ← siguiente paso
      user:      "Observation: ..."
      ...
      assistant: "Thought: ...\nFinal Answer: ..."                  ← cierre

    Formato de salida: {"messages": [...]} compatible con LLMTrainer.
    """
    system = _build_domestic_system()
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Tarea: {ex['user_request']}"},
    ]

    for step in ex["steps"]:
        # Turno assistant: razonamiento + llamada a herramienta
        assistant_text = (
            f"Thought: {step['thought']}\n"
            f"Action: {step['tool']}\n"
            f"Action Input: {json.dumps(step['args'], ensure_ascii=False)}"
        )
        messages.append({"role": "assistant", "content": assistant_text})

        # Turno user: Observation (resultado de la herramienta)
        messages.append({"role": "user", "content": f"Observation: {step['result']}"})

    # Turno final del assistant: razonamiento + respuesta
    final_text = f"Thought: {ex['final_thought']}\nFinal Answer: {ex['final_answer']}"
    messages.append({"role": "assistant", "content": final_text})

    return {"messages": messages}


# ---------------------------------------------------------------------------
# Templates base (~60 ejemplos de alta calidad)
# Distribución: file_organize(30%), email_filter(22%), calendar_get(18%),
#               note_save(13%), search_files(12%), process_run(5%)
# ---------------------------------------------------------------------------

_TEMPLATES: list[DomesticExample] = [

    # =========================================================================
    # file_organize  (18 templates)
    # =========================================================================

    # --- patrón: búsqueda + mover 1 tipo ---
    {
        "user_request": "Organiza las facturas del escritorio en una carpeta de Documentos",
        "steps": [
            {
                "thought": "Busco archivos con 'factura' en el escritorio para ver qué hay.",
                "tool": "search_files",
                "args": {"query": "factura", "path": "~/Desktop", "extensions": [".pdf", ".PDF"]},
                "result": "search_files: 2 coincidencia(s) en 2 archivo(s).\n  ~/Desktop/factura_ene.pdf  (línea 1)\n  ~/Desktop/factura_feb.pdf  (línea 1)",
            },
            {
                "thought": "Hay dos facturas PDF. Hago dry_run para confirmar el movimiento.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/factura_ene.pdf", "~/Desktop/factura_feb.pdf"], "dest": "~/Documentos/Facturas/", "dry_run": True},
                "result": "[dry_run] file_organize: 2 archivos procesados, 2 a mover, 0 errores. Pasa dry_run=False para ejecutar.\n  [dry_run] →  factura_ene.pdf  (124,532 B)  →  ~/Documentos/Facturas/\n  [dry_run] →  factura_feb.pdf  (98,441 B)  →  ~/Documentos/Facturas/",
            },
            {
                "thought": "El dry_run es correcto. Ejecuto el movimiento.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/factura_ene.pdf", "~/Desktop/factura_feb.pdf"], "dest": "~/Documentos/Facturas/", "dry_run": False},
                "result": "file_organize: 2 archivos procesados, 2 a mover, 0 errores.\n  ✓  factura_ene.pdf  →  ~/Documentos/Facturas/\n  ✓  factura_feb.pdf  →  ~/Documentos/Facturas/",
            },
        ],
        "final_thought": "Las dos facturas se han movido correctamente.",
        "final_answer": "Listo. He movido 2 facturas (factura_ene.pdf y factura_feb.pdf) desde el escritorio a ~/Documentos/Facturas/.",
        "language": "es",
        "category": "file_organize",
    },

    # --- patrón: mover directamente sin búsqueda ---
    {
        "user_request": "Mueve ~/Descargas/contrato_2026.pdf a ~/Documentos/Contratos/",
        "steps": [
            {
                "thought": "El usuario indica el archivo y el destino exactos. Hago dry_run primero.",
                "tool": "file_organize",
                "args": {"files": ["~/Descargas/contrato_2026.pdf"], "dest": "~/Documentos/Contratos/", "dry_run": True},
                "result": "[dry_run] file_organize: 1 archivo procesado, 1 a mover, 0 errores. Pasa dry_run=False para ejecutar.\n  [dry_run] →  contrato_2026.pdf  (245,120 B)  →  ~/Documentos/Contratos/",
            },
            {
                "thought": "Confirmado. Ejecuto el movimiento.",
                "tool": "file_organize",
                "args": {"files": ["~/Descargas/contrato_2026.pdf"], "dest": "~/Documentos/Contratos/", "dry_run": False},
                "result": "file_organize: 1 archivo procesado, 1 a mover, 0 errores.\n  ✓  contrato_2026.pdf  →  ~/Documentos/Contratos/",
            },
        ],
        "final_thought": "El archivo se movió sin errores.",
        "final_answer": "Hecho. contrato_2026.pdf está ahora en ~/Documentos/Contratos/.",
        "language": "es",
        "category": "file_organize",
    },

    # --- patrón: archivo no encontrado ---
    {
        "user_request": "Mueve el archivo presupuesto_final.xlsx del escritorio a ~/Documentos/",
        "steps": [
            {
                "thought": "Intento mover el archivo. Si no existe, el resultado lo indicará.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/presupuesto_final.xlsx"], "dest": "~/Documentos/", "dry_run": True},
                "result": "[dry_run] file_organize: 1 archivos procesados, 0 a mover, 1 errores. Pasa dry_run=False para ejecutar.\n  [dry_run] ✗  No encontrado: ~/Desktop/presupuesto_final.xlsx",
            },
            {
                "thought": "El archivo no existe en el escritorio. Busco con nombre similar.",
                "tool": "search_files",
                "args": {"query": "presupuesto", "path": "~/Desktop", "extensions": [".xlsx", ".xls", ".ods"]},
                "result": "search_files: ninguna coincidencia para 'presupuesto' en '~/Desktop' (0 archivos revisados).",
            },
        ],
        "final_thought": "El archivo no existe en el escritorio ni en formatos similares.",
        "final_answer": "No he encontrado 'presupuesto_final.xlsx' ni ningún archivo de presupuesto en el escritorio. ¿Puedes verificar la ubicación exacta?",
        "language": "es",
        "category": "file_organize",
    },

    # --- patrón: dry_run rechazado por el usuario (solo dry_run, sin ejecutar) ---
    {
        "user_request": "¿Qué pasaría si moviera todos los .mp3 de Descargas a ~/Música/?",
        "steps": [
            {
                "thought": "El usuario pide solo una previsualización (dry_run). Busco los MP3 primero.",
                "tool": "search_files",
                "args": {"query": "", "path": "~/Descargas", "extensions": [".mp3"]},
                "result": "search_files: 5 coincidencia(s) en 5 archivo(s).\n  ~/Descargas/cancion1.mp3\n  ~/Descargas/podcast_ep12.mp3\n  ~/Descargas/soundtrack.mp3\n  ~/Descargas/tema_live.mp3\n  ~/Descargas/remix_2026.mp3",
            },
            {
                "thought": "Ejecuto dry_run con todos los MP3 encontrados.",
                "tool": "file_organize",
                "args": {"files": ["~/Descargas/cancion1.mp3", "~/Descargas/podcast_ep12.mp3", "~/Descargas/soundtrack.mp3", "~/Descargas/tema_live.mp3", "~/Descargas/remix_2026.mp3"], "dest": "~/Música/", "dry_run": True},
                "result": "[dry_run] file_organize: 5 archivos procesados, 5 a mover, 0 errores. Pasa dry_run=False para ejecutar.\n  [dry_run] →  cancion1.mp3  →  ~/Música/\n  [dry_run] →  podcast_ep12.mp3  →  ~/Música/\n  [dry_run] →  soundtrack.mp3  →  ~/Música/\n  [dry_run] →  tema_live.mp3  →  ~/Música/\n  [dry_run] →  remix_2026.mp3  →  ~/Música/",
            },
        ],
        "final_thought": "El dry_run muestra exactamente qué se movería. Solo informo, no ejecuto.",
        "final_answer": "Si ejecutaras la operación, se moverían 5 archivos MP3 desde ~/Descargas/ a ~/Música/: cancion1.mp3, podcast_ep12.mp3, soundtrack.mp3, tema_live.mp3 y remix_2026.mp3. Ninguno tiene conflictos de nombre en el destino. Dime si quieres que lo haga.",
        "language": "es",
        "category": "file_organize",
    },

    # --- patrón: conflicto de nombre en destino ---
    {
        "user_request": "Mueve ~/Desktop/informe.pdf a ~/Documentos/Informes/",
        "steps": [
            {
                "thought": "Intento mover el archivo con dry_run.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/informe.pdf"], "dest": "~/Documentos/Informes/", "dry_run": True},
                "result": "[dry_run] file_organize: 1 archivos procesados, 0 a mover, 0 errores. Pasa dry_run=False para ejecutar.\n  [dry_run] ⚠  Ya existe en destino: informe.pdf (se omite)",
            },
        ],
        "final_thought": "El archivo ya existe en el destino. No se puede mover sin sobrescribir.",
        "final_answer": "No se ha movido el archivo: ya existe un 'informe.pdf' en ~/Documentos/Informes/. Para no perder el original, renombra uno de los dos primero. ¿Quieres que te ayude con eso?",
        "language": "es",
        "category": "file_organize",
    },

    # --- patrón: múltiples destinos (dos llamadas) ---
    {
        "user_request": "Organiza el escritorio: los PDFs a Documentos y las fotos JPG a Imágenes",
        "steps": [
            {
                "thought": "Busco PDFs en el escritorio.",
                "tool": "search_files",
                "args": {"query": "", "path": "~/Desktop", "extensions": [".pdf"]},
                "result": "search_files: 3 coincidencia(s) en 3 archivo(s).\n  ~/Desktop/contrato.pdf\n  ~/Desktop/recibo_luz.pdf\n  ~/Desktop/seguro.pdf",
            },
            {
                "thought": "Muevo los PDFs con dry_run.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/contrato.pdf", "~/Desktop/recibo_luz.pdf", "~/Desktop/seguro.pdf"], "dest": "~/Documentos/", "dry_run": False},
                "result": "file_organize: 3 archivos procesados, 3 a mover, 0 errores.\n  ✓  contrato.pdf  →  ~/Documentos/\n  ✓  recibo_luz.pdf  →  ~/Documentos/\n  ✓  seguro.pdf  →  ~/Documentos/",
            },
            {
                "thought": "Ahora busco las fotos JPG.",
                "tool": "search_files",
                "args": {"query": "", "path": "~/Desktop", "extensions": [".jpg", ".jpeg"]},
                "result": "search_files: 4 coincidencia(s) en 4 archivo(s).\n  ~/Desktop/foto_cumple.jpg\n  ~/Desktop/vacaciones.jpg\n  ~/Desktop/perfil.jpg\n  ~/Desktop/captura.jpg",
            },
            {
                "thought": "Muevo las fotos a Imágenes.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/foto_cumple.jpg", "~/Desktop/vacaciones.jpg", "~/Desktop/perfil.jpg", "~/Desktop/captura.jpg"], "dest": "~/Imágenes/", "dry_run": False},
                "result": "file_organize: 4 archivos procesados, 4 a mover, 0 errores.\n  ✓  foto_cumple.jpg  →  ~/Imágenes/\n  ✓  vacaciones.jpg  →  ~/Imágenes/\n  ✓  perfil.jpg  →  ~/Imágenes/\n  ✓  captura.jpg  →  ~/Imágenes/",
            },
        ],
        "final_thought": "Todos los archivos se han organizado correctamente.",
        "final_answer": "Escritorio organizado: 3 PDFs movidos a ~/Documentos/ y 4 fotos JPG a ~/Imágenes/.",
        "language": "es",
        "category": "file_organize",
    },

    # --- English ---
    {
        "user_request": "Move all PDF files from Downloads to Documents/Reports",
        "steps": [
            {
                "thought": "I'll search for PDFs in the Downloads folder first.",
                "tool": "search_files",
                "args": {"query": "", "path": "~/Downloads", "extensions": [".pdf"]},
                "result": "search_files: 3 match(es) in 3 file(s).\n  ~/Downloads/report_q1.pdf\n  ~/Downloads/invoice_march.pdf\n  ~/Downloads/contract.pdf",
            },
            {
                "thought": "Found 3 PDFs. Moving them to Documents/Reports.",
                "tool": "file_organize",
                "args": {"files": ["~/Downloads/report_q1.pdf", "~/Downloads/invoice_march.pdf", "~/Downloads/contract.pdf"], "dest": "~/Documents/Reports/", "dry_run": False},
                "result": "file_organize: 3 files processed, 3 moved, 0 errors.\n  ✓  report_q1.pdf  →  ~/Documents/Reports/\n  ✓  invoice_march.pdf  →  ~/Documents/Reports/\n  ✓  contract.pdf  →  ~/Documents/Reports/",
            },
        ],
        "final_thought": "All PDFs have been moved successfully.",
        "final_answer": "Done. 3 PDF files (report_q1.pdf, invoice_march.pdf, contract.pdf) moved from ~/Downloads/ to ~/Documents/Reports/.",
        "language": "en",
        "category": "file_organize",
    },

    {
        "user_request": "Clean up my desktop by moving screenshots to Pictures",
        "steps": [
            {
                "thought": "I'll search for screenshot files on the desktop.",
                "tool": "search_files",
                "args": {"query": "screenshot", "path": "~/Desktop"},
                "result": "search_files: 5 match(es) in 5 file(s).\n  ~/Desktop/screenshot_2026-01-10.png\n  ~/Desktop/screenshot_2026-02-03.png\n  ~/Desktop/screenshot_meeting.png\n  ~/Desktop/screenshot_error.png\n  ~/Desktop/screenshot_code.png",
            },
            {
                "thought": "5 screenshots found. I'll do a dry run first.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/screenshot_2026-01-10.png", "~/Desktop/screenshot_2026-02-03.png", "~/Desktop/screenshot_meeting.png", "~/Desktop/screenshot_error.png", "~/Desktop/screenshot_code.png"], "dest": "~/Pictures/Screenshots/", "dry_run": True},
                "result": "[dry_run] file_organize: 5 files processed, 5 to move, 0 errors.\n  [dry_run] →  screenshot_2026-01-10.png  →  ~/Pictures/Screenshots/\n  [dry_run] →  screenshot_2026-02-03.png  →  ~/Pictures/Screenshots/\n  [dry_run] →  screenshot_meeting.png  →  ~/Pictures/Screenshots/\n  [dry_run] →  screenshot_error.png  →  ~/Pictures/Screenshots/\n  [dry_run] →  screenshot_code.png  →  ~/Pictures/Screenshots/",
            },
            {
                "thought": "Dry run looks correct. Executing the move.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/screenshot_2026-01-10.png", "~/Desktop/screenshot_2026-02-03.png", "~/Desktop/screenshot_meeting.png", "~/Desktop/screenshot_error.png", "~/Desktop/screenshot_code.png"], "dest": "~/Pictures/Screenshots/", "dry_run": False},
                "result": "file_organize: 5 files processed, 5 moved, 0 errors.",
            },
        ],
        "final_thought": "All screenshots moved successfully.",
        "final_answer": "Done! 5 screenshots moved from Desktop to ~/Pictures/Screenshots/.",
        "language": "en",
        "category": "file_organize",
    },

    # --- patrón: destino fuera de rutas seguras ---
    {
        "user_request": "Mueve ~/Desktop/config.txt a /etc/",
        "steps": [
            {
                "thought": "Intento mover el archivo al destino indicado.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/config.txt"], "dest": "/etc/", "dry_run": False},
                "result": "[Bloqueado]: el destino '/etc/' está fuera de las rutas seguras permitidas. Usa una carpeta dentro de Escritorio, Documentos, Descargas, Notas o similares.",
            },
        ],
        "final_thought": "La operación está bloqueada por seguridad.",
        "final_answer": "No puedo mover archivos a /etc/ porque es una carpeta del sistema protegida. Solo puedo mover archivos dentro de carpetas de usuario como ~/Documentos, ~/Desktop, ~/Descargas o ~/Notas.",
        "language": "es",
        "category": "file_organize",
    },

    # --- patrón: un solo archivo ya en destino correcto ---
    {
        "user_request": "Organiza los archivos de texto del escritorio en ~/Notas",
        "steps": [
            {
                "thought": "Busco archivos .txt en el escritorio.",
                "tool": "search_files",
                "args": {"query": "", "path": "~/Desktop", "extensions": [".txt", ".md"]},
                "result": "search_files: 2 coincidencia(s) en 2 archivo(s).\n  ~/Desktop/ideas.txt\n  ~/Desktop/recordatorio.md",
            },
            {
                "thought": "Muevo los archivos de texto a ~/Notas.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/ideas.txt", "~/Desktop/recordatorio.md"], "dest": "~/Notas/", "dry_run": False},
                "result": "file_organize: 2 archivos procesados, 2 a mover, 0 errores.\n  ✓  ideas.txt  →  ~/Notas/\n  ✓  recordatorio.md  →  ~/Notas/",
            },
        ],
        "final_thought": "Los archivos de texto se han movido a Notas.",
        "final_answer": "Hecho. He movido 2 archivos de texto (ideas.txt y recordatorio.md) del escritorio a ~/Notas/.",
        "language": "es",
        "category": "file_organize",
    },

    # =========================================================================
    # email_filter  (13 templates)
    # =========================================================================

    {
        "user_request": "Muéstrame los correos del banco en mi INBOX",
        "steps": [
            {
                "thought": "El usuario quiere ver correos del banco. Filtro por remitente.",
                "tool": "email_filter",
                "args": {"server": "imap.gmail.com", "user": "usuario@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "sender": "banco", "mode": "list", "limit": 10},
                "result": "email_filter: 4 mensajes encontrados en 'INBOX'\n  [1234] 14 mayo 2026  De: notificaciones@banco.es  Asunto: Movimiento en tu cuenta - 120,00 €\n  [1198] 10 mayo 2026  De: notificaciones@banco.es  Asunto: Tu extracto mensual está listo\n  [1145] 05 mayo 2026  De: notificaciones@banco.es  Asunto: Recordatorio: domiciliación recibida\n  [1102] 01 mayo 2026  De: alertas@banco.es  Asunto: Nuevo acceso a tu banca online",
            },
        ],
        "final_thought": "He obtenido los 4 correos del banco.",
        "final_answer": "Tienes 4 correos del banco en tu INBOX:\n1. (14 mayo) Movimiento en tu cuenta - 120,00 €\n2. (10 mayo) Tu extracto mensual está listo\n3. (5 mayo) Domiciliación recibida\n4. (1 mayo) Nuevo acceso a banca online",
        "language": "es",
        "category": "email_filter",
    },

    {
        "user_request": "¿Cuántos emails tengo de Amazon con asunto 'pedido'?",
        "steps": [
            {
                "thought": "Busco en INBOX emails de Amazon con 'pedido' en el asunto.",
                "tool": "email_filter",
                "args": {"server": "imap.gmail.com", "user": "usuario@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "sender": "amazon", "subject": "pedido", "mode": "list", "limit": 20},
                "result": "email_filter: 6 mensajes encontrados en 'INBOX'\n  [2341] 12 mayo 2026  De: auto-confirm@amazon.es  Asunto: Tu pedido #114-2938471 ha sido enviado\n  [2298] 08 mayo 2026  De: auto-confirm@amazon.es  Asunto: Pedido entregado: #114-2847293\n  [2201] 02 mayo 2026  De: auto-confirm@amazon.es  Asunto: Confirmación de pedido #114-2819823\n  [2150] 28 abr 2026  De: auto-confirm@amazon.es  Asunto: Tu pedido #114-2744182 llegará mañana\n  [2089] 20 abr 2026  De: auto-confirm@amazon.es  Asunto: Pedido devuelto: #114-2701234\n  [2034] 15 abr 2026  De: auto-confirm@amazon.es  Asunto: Tu pedido #114-2671928 está en camino\n  ... y 0 mensajes más",
            },
        ],
        "final_thought": "El resultado muestra 6 emails de Amazon sobre pedidos.",
        "final_answer": "Tienes 6 correos de Amazon con 'pedido' en el asunto. Los más recientes: un envío del 12 mayo, una entrega del 8 mayo y una confirmación del 2 mayo.",
        "language": "es",
        "category": "email_filter",
    },

    {
        "user_request": "Lista los últimos 5 correos no leídos de mi jefa (ana.garcia@empresa.com)",
        "steps": [
            {
                "thought": "Filtro por remitente ana.garcia@empresa.com, límite 5.",
                "tool": "email_filter",
                "args": {"server": "imap.gmail.com", "user": "usuario@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "sender": "ana.garcia@empresa.com", "mode": "list", "limit": 5},
                "result": "email_filter: 3 mensajes encontrados en 'INBOX'\n  [3421] 14 mayo 2026  De: ana.garcia@empresa.com  Asunto: Revisión del informe trimestral\n  [3387] 13 mayo 2026  De: ana.garcia@empresa.com  Asunto: Reunión mañana a las 10h\n  [3312] 09 mayo 2026  De: ana.garcia@empresa.com  Asunto: FWD: Presupuesto Q2 aprobado",
            },
        ],
        "final_thought": "Encontré 3 correos recientes de ana.garcia@empresa.com.",
        "final_answer": "Tienes 3 correos recientes de Ana García:\n1. (14 mayo) Revisión del informe trimestral\n2. (13 mayo) Reunión mañana a las 10h\n3. (9 mayo) Presupuesto Q2 aprobado",
        "language": "es",
        "category": "email_filter",
    },

    {
        "user_request": "Marca como spam los correos de 'noreply@promo-ofertas.net'",
        "steps": [
            {
                "thought": "Primero listo los correos de ese remitente para ver cuántos son.",
                "tool": "email_filter",
                "args": {"server": "imap.gmail.com", "user": "usuario@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "sender": "noreply@promo-ofertas.net", "mode": "list", "limit": 20},
                "result": "email_filter: 8 mensajes encontrados en 'INBOX'\n  [891] 14 mayo 2026  De: noreply@promo-ofertas.net  Asunto: ¡OFERTA EXCLUSIVA solo hoy!\n  [832] 13 mayo 2026  De: noreply@promo-ofertas.net  Asunto: Tu premio está esperando\n  ... y 6 mensajes más",
            },
            {
                "thought": "8 correos spam. Ejecuto mark_spam con dry_run primero.",
                "tool": "email_filter",
                "args": {"server": "imap.gmail.com", "user": "usuario@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "sender": "noreply@promo-ofertas.net", "mode": "mark_spam", "limit": 20, "dry_run": True},
                "result": "[dry_run] email_filter modo='mark_spam' — no se modificará nada. email_filter: 8 mensajes encontrados en 'INBOX' — acción 'mark_spam' se ejecutaría sobre 8 mensajes.",
            },
            {
                "thought": "El dry_run muestra 8 mensajes a marcar. Ejecuto.",
                "tool": "email_filter",
                "args": {"server": "imap.gmail.com", "user": "usuario@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "sender": "noreply@promo-ofertas.net", "mode": "mark_spam", "limit": 20, "dry_run": False},
                "result": "email_filter: 8 mensajes encontrados en 'INBOX' — acción 'mark_spam' pendiente de implementación.",
            },
        ],
        "final_thought": "La acción está en proceso de implementación, informo al usuario.",
        "final_answer": "He identificado 8 correos de noreply@promo-ofertas.net. La función de marcar como spam está disponible pero aún en implementación completa. Por ahora puedo listarlos para que los elimines manualmente.",
        "language": "es",
        "category": "email_filter",
    },

    {
        "user_request": "¿Hay algún correo de mi banco con asunto 'extracto' este mes?",
        "steps": [
            {
                "thought": "Busco correos con 'extracto' en el asunto de remitentes del banco.",
                "tool": "email_filter",
                "args": {"server": "imap.gmail.com", "user": "usuario@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "subject": "extracto", "mode": "list", "limit": 5},
                "result": "email_filter: 1 mensajes encontrados en 'INBOX'\n  [1198] 10 mayo 2026  De: notificaciones@banco.es  Asunto: Tu extracto mensual de abril está listo",
            },
        ],
        "final_thought": "Hay un correo con extracto del mes de abril.",
        "final_answer": "Sí, tienes 1 correo con extracto: del 10 de mayo, enviado por notificaciones@banco.es con el extracto mensual de abril.",
        "language": "es",
        "category": "email_filter",
    },

    {
        "user_request": "No encuentro el correo de confirmación de vuelo de Iberia",
        "steps": [
            {
                "thought": "Busco en INBOX correos de Iberia.",
                "tool": "email_filter",
                "args": {"server": "imap.gmail.com", "user": "usuario@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "sender": "iberia", "mode": "list", "limit": 10},
                "result": "email_filter: 0 mensajes encontrados en 'INBOX'.",
            },
            {
                "thought": "No está en INBOX. Pruebo en la carpeta de spam.",
                "tool": "email_filter",
                "args": {"server": "imap.gmail.com", "user": "usuario@gmail.com", "password": "app-pass-1234", "folder": "Spam", "sender": "iberia", "mode": "list", "limit": 10},
                "result": "email_filter: 1 mensajes encontrados en 'Spam'\n  [44] 12 mayo 2026  De: confirmar@iberia.com  Asunto: Confirmación de reserva IB2341 MAD-BCN 20 mayo",
            },
        ],
        "final_thought": "El correo de Iberia estaba en la carpeta de spam.",
        "final_answer": "Encontré el correo de Iberia en tu carpeta de Spam: confirmación de reserva IB2341 MAD-BCN para el 20 de mayo. Te recomiendo moverlo a la bandeja de entrada.",
        "language": "es",
        "category": "email_filter",
    },

    # --- English ---
    {
        "user_request": "Show me emails from GitHub notifications",
        "steps": [
            {
                "thought": "I'll filter emails by sender 'github' in the inbox.",
                "tool": "email_filter",
                "args": {"server": "imap.gmail.com", "user": "user@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "sender": "github", "mode": "list", "limit": 10},
                "result": "email_filter: 5 messages found in 'INBOX'\n  [5621] 14 May 2026  From: notifications@github.com  Subject: [repo] Pull request #42 merged\n  [5598] 13 May 2026  From: notifications@github.com  Subject: [repo] New issue opened: Bug in login\n  [5543] 11 May 2026  From: notifications@github.com  Subject: [repo] Your PR was reviewed\n  [5490] 08 May 2026  From: noreply@github.com  Subject: Security alert: dependency vulnerability\n  [5412] 05 May 2026  From: notifications@github.com  Subject: [repo] Comment on issue #38",
            },
        ],
        "final_thought": "Found 5 GitHub notification emails.",
        "final_answer": "You have 5 GitHub notification emails. Most recent: PR #42 merged (14 May), new issue about login bug (13 May), and a security alert about a dependency vulnerability (8 May) that might need attention.",
        "language": "en",
        "category": "email_filter",
    },

    # =========================================================================
    # calendar_get  (11 templates)
    # =========================================================================

    {
        "user_request": "¿Qué tengo hoy en el calendario?",
        "steps": [
            {
                "thought": "Consulto los eventos de hoy.",
                "tool": "calendar_get",
                "args": {"date_range": "today"},
                "result": "calendar_get (today): 2 evento(s) entre 2026-05-14 y 2026-05-14.\n  2026-05-14  Reunión con el equipo de desarrollo  @Sala B\n  2026-05-14  Llamada con cliente - Proyecto V3",
            },
        ],
        "final_thought": "Hay dos eventos hoy.",
        "final_answer": "Hoy tienes 2 eventos:\n1. Reunión con el equipo de desarrollo en Sala B\n2. Llamada con cliente - Proyecto V3",
        "language": "es",
        "category": "calendar_get",
    },

    {
        "user_request": "¿Tengo algo importante esta semana?",
        "steps": [
            {
                "thought": "Consulto los eventos de la semana.",
                "tool": "calendar_get",
                "args": {"date_range": "week"},
                "result": "calendar_get (week): 5 evento(s) entre 2026-05-14 y 2026-05-20.\n  2026-05-14  Reunión de equipo\n  2026-05-15  Dentista  @Clínica Dr. López\n  2026-05-16  Entrega informe Q2\n  2026-05-19  Cumpleaños Ana\n  2026-05-20  Revisión de presupuesto anual  @Sala de juntas",
            },
        ],
        "final_thought": "Hay 5 eventos esta semana, incluyendo la entrega del informe Q2 el viernes.",
        "final_answer": "Esta semana tienes 5 eventos:\n• Hoy: Reunión de equipo\n• Mañana (15 mayo): Dentista\n• Sábado 16: Entrega del informe Q2 (importante)\n• Lunes 19: Cumpleaños de Ana\n• Martes 20: Revisión de presupuesto anual",
        "language": "es",
        "category": "calendar_get",
    },

    {
        "user_request": "¿Qué eventos tengo el 20 de mayo?",
        "steps": [
            {
                "thought": "Consulto los eventos del 20 de mayo de 2026.",
                "tool": "calendar_get",
                "args": {"date_range": "2026-05-20"},
                "result": "calendar_get (2026-05-20): 2 evento(s) entre 2026-05-20 y 2026-05-20.\n  2026-05-20  Revisión de presupuesto anual  @Sala de juntas\n  2026-05-20  Vuelo MAD-BCN  @Aeropuerto Barajas T4",
            },
        ],
        "final_thought": "Hay dos eventos el 20 de mayo.",
        "final_answer": "El 20 de mayo tienes:\n1. Revisión de presupuesto anual en Sala de juntas\n2. Vuelo MAD-BCN desde T4 (Barajas)",
        "language": "es",
        "category": "calendar_get",
    },

    {
        "user_request": "Dime los eventos de este mes",
        "steps": [
            {
                "thought": "Consulto los eventos del mes.",
                "tool": "calendar_get",
                "args": {"date_range": "month"},
                "result": "calendar_get (month): 8 evento(s) entre 2026-05-14 y 2026-06-13.\n  2026-05-14  Reunión de equipo\n  2026-05-15  Dentista\n  2026-05-20  Vuelo MAD-BCN\n  2026-05-23  Conferencia DevFest  @Madrid\n  2026-05-28  Revisión trimestral\n  2026-06-01  Inicio periodo vacaciones\n  2026-06-05  Cumpleaños papá\n  2026-06-10  Revisión médica anual",
            },
        ],
        "final_thought": "Hay 8 eventos en los próximos 30 días.",
        "final_answer": "En los próximos 30 días tienes 8 eventos. Los más destacados: vuelo MAD-BCN el 20 mayo, DevFest en Madrid el 23 mayo, y el inicio de vacaciones el 1 de junio.",
        "language": "es",
        "category": "calendar_get",
    },

    {
        "user_request": "¿Tengo eventos entre el 1 y el 5 de junio?",
        "steps": [
            {
                "thought": "Consulto el rango de fechas del 1 al 5 de junio.",
                "tool": "calendar_get",
                "args": {"date_range": "2026-06-01/2026-06-05"},
                "result": "calendar_get (2026-06-01/2026-06-05): 2 evento(s) entre 2026-06-01 y 2026-06-05.\n  2026-06-01  Inicio periodo vacaciones\n  2026-06-05  Cumpleaños papá",
            },
        ],
        "final_thought": "Hay dos eventos en ese rango.",
        "final_answer": "Del 1 al 5 de junio tienes: inicio de vacaciones el 1 junio y el cumpleaños de papá el 5 junio.",
        "language": "es",
        "category": "calendar_get",
    },

    {
        "user_request": "¿No tengo nada en el calendario hoy?",
        "steps": [
            {
                "thought": "Consulto los eventos de hoy para confirmar.",
                "tool": "calendar_get",
                "args": {"date_range": "today"},
                "result": "calendar_get (today): No se encontraron eventos entre 2026-05-14 y 2026-05-14.",
            },
        ],
        "final_thought": "El calendario no tiene eventos para hoy.",
        "final_answer": "Efectivamente, hoy no tienes ningún evento en el calendario.",
        "language": "es",
        "category": "calendar_get",
    },

    # --- English ---
    {
        "user_request": "What's on my calendar for this week?",
        "steps": [
            {
                "thought": "I'll check the calendar for the current week.",
                "tool": "calendar_get",
                "args": {"date_range": "week"},
                "result": "calendar_get (week): 4 event(s) between 2026-05-14 and 2026-05-20.\n  2026-05-14  Team standup  @Conference room\n  2026-05-15  Doctor appointment  @City clinic\n  2026-05-18  Project deadline: API v2\n  2026-05-20  Sprint review and retrospective",
            },
        ],
        "final_thought": "There are 4 events this week.",
        "final_answer": "This week you have 4 events: team standup today, doctor appointment tomorrow (15th), API v2 deadline on Monday 18th, and sprint review on Wednesday 20th.",
        "language": "en",
        "category": "calendar_get",
    },

    # =========================================================================
    # note_save  (8 templates)
    # =========================================================================

    {
        "user_request": "Guarda una nota: 'Llamar al seguro mañana antes de las 9h'",
        "steps": [
            {
                "thought": "Guardo la nota con el texto indicado.",
                "tool": "note_save",
                "args": {"title": "Llamar al seguro", "body": "Llamar al seguro mañana antes de las 9h"},
                "result": "note_save: nota guardada en '~/Notas/20260514_143022_Llamar al seguro.txt' (78 caracteres).",
            },
        ],
        "final_thought": "La nota se ha guardado correctamente.",
        "final_answer": "Nota guardada en ~/Notas/. Título: 'Llamar al seguro'. Recuerda: llamar antes de las 9h de mañana.",
        "language": "es",
        "category": "note_save",
    },

    {
        "user_request": "Toma nota de la reunión: asistieron Carlos, María y Lucía. Decisión: lanzar v2 el 1 de junio.",
        "steps": [
            {
                "thought": "Guardo las notas de la reunión con título descriptivo.",
                "tool": "note_save",
                "args": {"title": "Acta reunión 14 mayo 2026", "body": "Asistentes: Carlos, María, Lucía.\n\nDecisión principal: lanzar v2 el 1 de junio de 2026.\n\nPróximos pasos:\n- Carlos: preparar despliegue\n- María: actualizar documentación\n- Lucía: coordinar QA"},
                "result": "note_save: nota guardada en '~/Notas/20260514_151234_Acta reunión 14 mayo 2026.txt' (212 caracteres).",
            },
        ],
        "final_thought": "El acta de la reunión se ha guardado.",
        "final_answer": "Acta guardada en ~/Notas/. Incluye asistentes (Carlos, María, Lucía), la decisión de lanzar v2 el 1 de junio y los próximos pasos asignados.",
        "language": "es",
        "category": "note_save",
    },

    {
        "user_request": "Guarda esto en mis notas: ideas para el proyecto — 1) usar Redis para cache 2) migrar a PostgreSQL 3) añadir tests de integración",
        "steps": [
            {
                "thought": "Guardo las ideas del proyecto como nota.",
                "tool": "note_save",
                "args": {"title": "Ideas proyecto", "body": "1) Usar Redis para cache\n2) Migrar a PostgreSQL\n3) Añadir tests de integración"},
                "result": "note_save: nota guardada en '~/Notas/20260514_162001_Ideas proyecto.txt' (98 caracteres).",
            },
        ],
        "final_thought": "Las ideas se guardaron correctamente.",
        "final_answer": "Nota guardada con las 3 ideas del proyecto: Redis para cache, migración a PostgreSQL y tests de integración.",
        "language": "es",
        "category": "note_save",
    },

    {
        "user_request": "Guarda una nota de compras: leche, pan, huevos, aceite de oliva, café",
        "steps": [
            {
                "thought": "Guardo la lista de la compra como nota.",
                "tool": "note_save",
                "args": {"title": "Lista de la compra", "body": "- Leche\n- Pan\n- Huevos\n- Aceite de oliva\n- Café"},
                "result": "note_save: nota guardada en '~/Notas/20260514_170344_Lista de la compra.txt' (60 caracteres).",
            },
        ],
        "final_thought": "La lista de compras se guardó.",
        "final_answer": "Lista guardada en ~/Notas/: leche, pan, huevos, aceite de oliva y café.",
        "language": "es",
        "category": "note_save",
    },

    {
        "user_request": "Guarda una nota en ~/Documentos/Trabajo/ con título 'OKRs Q2' y el contenido: Objetivo 1: lanzar v2. Objetivo 2: reducir bugs un 30%.",
        "steps": [
            {
                "thought": "Guardo la nota en la carpeta especificada por el usuario.",
                "tool": "note_save",
                "args": {"title": "OKRs Q2", "body": "Objetivo 1: lanzar v2.\nObjetivo 2: reducir bugs un 30%.", "folder": "~/Documentos/Trabajo/"},
                "result": "note_save: nota guardada en 'C:/Users/usuario/Documentos/Trabajo/20260514_180012_OKRs Q2.txt' (72 caracteres).",
            },
        ],
        "final_thought": "La nota de OKRs se guardó en la carpeta indicada.",
        "final_answer": "Nota guardada en ~/Documentos/Trabajo/OKRs Q2 con los dos objetivos del Q2.",
        "language": "es",
        "category": "note_save",
    },

    # --- English ---
    {
        "user_request": "Save a note: remember to renew passport before July",
        "steps": [
            {
                "thought": "Saving the reminder note.",
                "tool": "note_save",
                "args": {"title": "Renew passport", "body": "Remember to renew passport before July 2026. Check requirements at embassy website."},
                "result": "note_save: note saved at '~/Notes/20260514_155023_Renew passport.txt' (96 characters).",
            },
        ],
        "final_thought": "The note was saved successfully.",
        "final_answer": "Note saved: 'Renew passport before July'. It's in ~/Notes/.",
        "language": "en",
        "category": "note_save",
    },

    # =========================================================================
    # search_files  (7 templates)
    # =========================================================================

    {
        "user_request": "Busca en mis documentos cualquier archivo que mencione 'contrato Telefónica'",
        "steps": [
            {
                "thought": "Busco la frase en los archivos de Documentos.",
                "tool": "search_files",
                "args": {"query": "contrato Telefónica", "path": "~/Documentos", "extensions": [".pdf", ".txt", ".docx", ".md"]},
                "result": "search_files: 1 coincidencia(s) en 1 archivo(s) (45 revisados).\n  ~/Documentos/contratos/telefonica_fibra_2024.txt  (línea 3)\n       2: \n    →  3: CONTRATO DE SERVICIO TELEFÓNICA FIBRA ÓPTICA\n       4: Fecha: 15 enero 2024",
            },
        ],
        "final_thought": "El archivo existe en ~/Documentos/contratos/.",
        "final_answer": "Encontré el contrato de Telefónica en ~/Documentos/contratos/telefonica_fibra_2024.txt (línea 3). Es un contrato de fibra óptica del 15 de enero de 2024.",
        "language": "es",
        "category": "search_files",
    },

    {
        "user_request": "¿En qué archivos Python tengo la función 'calcular_iva'?",
        "steps": [
            {
                "thought": "Busco la función en archivos Python.",
                "tool": "search_files",
                "args": {"query": "calcular_iva", "path": "~/Desktop/Proyecto_V3", "extensions": [".py"]},
                "result": "search_files: 2 coincidencia(s) en 2 archivo(s) (12 revisados).\n  ~/Desktop/Proyecto_V3/utils.py  (línea 47)\n    →  47: def calcular_iva(precio, tipo=0.21):\n  ~/Desktop/Proyecto_V3/tests/test_utils.py  (línea 23)\n    →  23: assert calcular_iva(100) == 121.0",
            },
        ],
        "final_thought": "La función existe en dos archivos.",
        "final_answer": "La función 'calcular_iva' aparece en 2 archivos: definida en utils.py (línea 47) y referenciada en test_utils.py (línea 23).",
        "language": "es",
        "category": "search_files",
    },

    {
        "user_request": "Busca si tengo alguna nota con la palabra 'contraseña'",
        "steps": [
            {
                "thought": "Busco 'contraseña' en la carpeta de notas.",
                "tool": "search_files",
                "args": {"query": "contraseña", "path": "~/Notas", "extensions": [".txt", ".md"]},
                "result": "search_files: 1 coincidencia(s) en 1 archivo(s) (8 revisados).\n  ~/Notas/20260102_accesos_wifi.txt  (línea 2)\n       1: Accesos red doméstica\n    →  2: contraseña wifi: ****",
            },
        ],
        "final_thought": "Hay una nota con esa palabra.",
        "final_answer": "Encontré una nota en ~/Notas/20260102_accesos_wifi.txt que menciona 'contraseña' (accesos de red doméstica). Considera eliminar contraseñas en texto plano.",
        "language": "es",
        "category": "search_files",
    },

    {
        "user_request": "Encuentra todos los archivos Markdown que hablen de 'sprint' en el proyecto",
        "steps": [
            {
                "thought": "Busco 'sprint' en archivos .md del proyecto.",
                "tool": "search_files",
                "args": {"query": "sprint", "path": "~/Desktop/Proyecto_V3", "extensions": [".md"], "max_results": 10},
                "result": "search_files: 10+ coincidencia(s) en 3 archivo(s) (5 revisados).\n  ~/Desktop/Proyecto_V3/README.md  (línea 480)\n  ~/Desktop/Proyecto_V3/SESION.md  (línea 12)\n  ~/Desktop/Proyecto_V3/GUIA_SERVIDOR.md  (línea 1)\n  [truncado a 10 resultados]",
            },
        ],
        "final_thought": "La palabra 'sprint' aparece en 3 archivos Markdown del proyecto.",
        "final_answer": "La palabra 'sprint' aparece en 3 archivos Markdown del proyecto: README.md, SESION.md y GUIA_SERVIDOR.md.",
        "language": "es",
        "category": "search_files",
    },

    {
        "user_request": "¿Tengo algún archivo de presupuesto en Descargas?",
        "steps": [
            {
                "thought": "Busco archivos relacionados con presupuesto en Descargas.",
                "tool": "search_files",
                "args": {"query": "presupuesto", "path": "~/Descargas"},
                "result": "search_files: ninguna coincidencia para 'presupuesto' en '~/Descargas' (34 archivos revisados).",
            },
        ],
        "final_thought": "No hay archivos de presupuesto en Descargas.",
        "final_answer": "No encontré ningún archivo que mencione 'presupuesto' en ~/Descargas (revisé 34 archivos). ¿Quieres que busque en otra carpeta?",
        "language": "es",
        "category": "search_files",
    },

    # --- English ---
    {
        "user_request": "Find all Python files in my project that import 'pandas'",
        "steps": [
            {
                "thought": "Searching for 'import pandas' or 'from pandas' in Python files.",
                "tool": "search_files",
                "args": {"query": "import pandas", "path": "~/Desktop/my_project", "extensions": [".py"]},
                "result": "search_files: 4 match(es) in 4 file(s) (18 reviewed).\n  ~/Desktop/my_project/analysis.py  (line 3)\n    →  3: import pandas as pd\n  ~/Desktop/my_project/data_loader.py  (line 1)\n    →  1: import pandas as pd\n  ~/Desktop/my_project/charts.py  (line 2)\n    →  2: from pandas import DataFrame\n  ~/Desktop/my_project/tests/test_analysis.py  (line 5)\n    →  5: import pandas as pd",
            },
        ],
        "final_thought": "Found 4 Python files using pandas.",
        "final_answer": "4 Python files import pandas: analysis.py, data_loader.py, charts.py (uses 'from pandas import DataFrame'), and tests/test_analysis.py.",
        "language": "en",
        "category": "search_files",
    },

    # =========================================================================
    # process_run  (3 templates)
    # =========================================================================

    {
        "user_request": "Comprueba si hay conexión a internet haciendo ping a localhost",
        "steps": [
            {
                "thought": "Ejecuto ping a localhost para comprobar la red.",
                "tool": "process_run",
                "args": {"name": "ping", "args": ["localhost", "-n", "3"], "dry_run": False},
                "result": "process_run: 'ping localhost -n 3' → código 0\nPing statistics for localhost:\n  Packets: Sent = 3, Received = 3, Lost = 0 (0% loss)",
            },
        ],
        "final_thought": "El ping a localhost fue exitoso, la conexión local funciona.",
        "final_answer": "Conexión local OK: ping a localhost con 3 paquetes enviados y 0 perdidos (0% pérdida).",
        "language": "es",
        "category": "process_run",
    },

    {
        "user_request": "¿Cuál es la IP de mi máquina?",
        "steps": [
            {
                "thought": "Uso ipconfig para obtener la información de red.",
                "tool": "process_run",
                "args": {"name": "ipconfig", "args": [], "dry_run": False},
                "result": "process_run: 'ipconfig' → código 0\nEthernet adapter:\n  IPv4 Address. . . : 192.168.1.100\n  Subnet Mask . . . : 255.255.255.0\n  Default Gateway . : 192.168.1.1",
            },
        ],
        "final_thought": "Tengo la información de red del usuario.",
        "final_answer": "Tu dirección IP local es 192.168.1.100, con máscara 255.255.255.0 y puerta de enlace 192.168.1.1.",
        "language": "es",
        "category": "process_run",
    },

    {
        "user_request": "Abre el explorador de archivos",
        "steps": [
            {
                "thought": "Ejecuto el explorador de archivos. Es un proceso de la whitelist.",
                "tool": "process_run",
                "args": {"name": "explorador", "args": [], "dry_run": False},
                "result": "process_run: 'explorer.exe' → código 0\n(sin salida)",
            },
        ],
        "final_thought": "El explorador se abrió.",
        "final_answer": "El explorador de archivos se ha abierto correctamente.",
        "language": "es",
        "category": "process_run",
    },

    # =========================================================================
    # DESCUBRIMIENTO DE RUTAS — process_run(ls/dir/find) + search_files
    # El agente lista el directorio o busca antes de asumir una ruta.
    # Estos ejemplos enseñan el patrón: explorar → localizar → actuar.
    # =========================================================================

    # --- ls ~ para descubrir estructura ---
    {
        "user_request": "Mueve el archivo de presupuesto del año pasado a ~/Documentos/Archivados/",
        "steps": [
            {
                "thought": "No sé dónde está el archivo exactamente. Primero listo el home para orientarme.",
                "tool": "process_run",
                "args": {"name": "ls", "args": ["~"], "dry_run": False},
                "result": "process_run: 'ls ~' → código 0\nDescargas  Desktop  Documentos  Imágenes  Música  Notas  Proyectos  Videos",
            },
            {
                "thought": "Hay una carpeta Documentos. Busco archivos de presupuesto allí.",
                "tool": "search_files",
                "args": {"query": "presupuesto", "path": "~", "extensions": [".xlsx", ".xls", ".ods", ".pdf"]},
                "result": "search_files: 2 coincidencia(s) en 2 archivo(s).\n  ~/Documentos/presupuesto_2025.xlsx  (línea 1)\n  ~/Desktop/presupuesto_draft.xlsx  (línea 1)",
            },
            {
                "thought": "El archivo del año pasado es presupuesto_2025.xlsx. Lo muevo a Archivados con dry_run.",
                "tool": "file_organize",
                "args": {"files": ["~/Documentos/presupuesto_2025.xlsx"], "dest": "~/Documentos/Archivados/", "dry_run": True},
                "result": "[dry_run] file_organize: 1 archivo procesado, 1 a mover, 0 errores.\n  [dry_run] →  presupuesto_2025.xlsx  →  ~/Documentos/Archivados/",
            },
            {
                "thought": "Dry run correcto. Ejecuto el movimiento.",
                "tool": "file_organize",
                "args": {"files": ["~/Documentos/presupuesto_2025.xlsx"], "dest": "~/Documentos/Archivados/", "dry_run": False},
                "result": "file_organize: 1 archivo procesado, 1 a mover, 0 errores.\n  ✓  presupuesto_2025.xlsx  →  ~/Documentos/Archivados/",
            },
        ],
        "final_thought": "El archivo de presupuesto 2025 está ahora en Archivados.",
        "final_answer": "Hecho. He movido presupuesto_2025.xlsx a ~/Documentos/Archivados/. El draft del escritorio no se ha tocado.",
        "language": "es",
        "category": "file_organize",
    },

    # --- dir para descubrir en Windows ---
    {
        "user_request": "¿Qué hay en mi carpeta de Descargas?",
        "steps": [
            {
                "thought": "Listo el contenido de Descargas con dir.",
                "tool": "process_run",
                "args": {"name": "dir", "args": ["~/Descargas"], "dry_run": False},
                "result": "process_run: 'dir ~/Descargas' → código 0\n Directorio de C:\\Users\\usuario\\Downloads\n\n15/05/2026  10:12    <DIR>          .\n15/05/2026  10:12    <DIR>          ..\n10/05/2026  09:45        1.234.567  proyecto_backup.zip\n12/05/2026  14:23          456.789  informe_mayo.pdf\n14/05/2026  16:00           78.000  notas_reunión.docx\n               3 archivos  1.769.356 bytes",
            },
        ],
        "final_thought": "La carpeta Descargas tiene 3 archivos.",
        "final_answer": "En ~/Descargas tienes 3 archivos: proyecto_backup.zip (1.2 MB, del 10 mayo), informe_mayo.pdf (456 KB, del 12 mayo) y notas_reunión.docx (78 KB, del 14 mayo).",
        "language": "es",
        "category": "process_run",
    },

    # --- find con -iname (case-insensitive) ---
    {
        "user_request": "Busca el archivo 'Informe_Final.pdf' en mi home, no sé dónde lo guardé",
        "steps": [
            {
                "thought": "Uso find con -iname para buscar sin distinguir mayúsculas/minúsculas.",
                "tool": "process_run",
                "args": {"name": "find", "args": ["~", "-iname", "informe_final.pdf"], "dry_run": False},
                "result": "process_run: 'find ~ -iname informe_final.pdf' → código 0\n~/Documentos/Trabajo/Informe_Final.pdf",
            },
        ],
        "final_thought": "El archivo está en ~/Documentos/Trabajo/.",
        "final_answer": "Encontré 'Informe_Final.pdf' en ~/Documentos/Trabajo/.",
        "language": "es",
        "category": "process_run",
    },

    # --- find para listar por extensión ---
    {
        "user_request": "¿Cuántos archivos PDF tengo en todo el home?",
        "steps": [
            {
                "thought": "Uso find para contar todos los PDFs del directorio home.",
                "tool": "process_run",
                "args": {"name": "find", "args": ["~", "-iname", "*.pdf"], "dry_run": False},
                "result": "process_run: 'find ~ -iname *.pdf' → código 0\n~/Documentos/contrato_fibra.pdf\n~/Documentos/Archivados/presupuesto_2024.pdf\n~/Desktop/factura_luz.pdf\n~/Descargas/informe_mayo.pdf\n~/Documentos/Trabajo/Informe_Final.pdf",
            },
        ],
        "final_thought": "Hay 5 PDFs en total en el home.",
        "final_answer": "Tienes 5 archivos PDF en tu home: contrato_fibra.pdf, presupuesto_2024.pdf (Archivados), factura_luz.pdf (Desktop), informe_mayo.pdf (Descargas) e Informe_Final.pdf (Trabajo).",
        "language": "es",
        "category": "process_run",
    },

    # --- discovery + reintento tras fallo ---
    {
        "user_request": "Organiza los archivos .log de mi proyecto en ~/Logs/",
        "steps": [
            {
                "thought": "Intento mover logs desde la ruta que parece más probable.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/app.log"], "dest": "~/Logs/", "dry_run": True},
                "result": "[dry_run] file_organize: 1 archivos procesados, 0 a mover, 1 errores.\n  [dry_run] ✗  No encontrado: ~/Desktop/app.log",
            },
            {
                "thought": "No está en el escritorio. Busco todos los .log en el home.",
                "tool": "search_files",
                "args": {"query": "", "path": "~", "extensions": [".log"]},
                "result": "search_files: 3 coincidencia(s) en 3 archivo(s).\n  ~/Proyecto_V3/app.log  (línea 1)\n  ~/Proyecto_V3/error.log  (línea 1)\n  ~/Proyecto_V3/train.log  (línea 1)",
            },
            {
                "thought": "Los logs están en ~/Proyecto_V3/. Los muevo a ~/Logs/.",
                "tool": "file_organize",
                "args": {"files": ["~/Proyecto_V3/app.log", "~/Proyecto_V3/error.log", "~/Proyecto_V3/train.log"], "dest": "~/Logs/", "dry_run": False},
                "result": "file_organize: 3 archivos procesados, 3 a mover, 0 errores.\n  ✓  app.log  →  ~/Logs/\n  ✓  error.log  →  ~/Logs/\n  ✓  train.log  →  ~/Logs/",
            },
        ],
        "final_thought": "Los 3 logs han sido movidos a ~/Logs/.",
        "final_answer": "Hecho. Encontré 3 archivos .log en ~/Proyecto_V3/ y los moví a ~/Logs/: app.log, error.log y train.log.",
        "language": "es",
        "category": "file_organize",
    },

    # --- discovery en inglés ---
    {
        "user_request": "I need to find my resume and move it to ~/Documents/Job/",
        "steps": [
            {
                "thought": "I don't know the exact location. Let me search for resume files.",
                "tool": "search_files",
                "args": {"query": "resume", "path": "~", "extensions": [".pdf", ".docx", ".doc"]},
                "result": "search_files: 2 match(es) in 2 file(s).\n  ~/Downloads/CV_John_2025.pdf  (line 1)\n  ~/Desktop/resume_draft.docx  (line 1)",
            },
            {
                "thought": "The CV_John_2025.pdf looks like the main resume. Moving it.",
                "tool": "file_organize",
                "args": {"files": ["~/Downloads/CV_John_2025.pdf"], "dest": "~/Documents/Job/", "dry_run": False},
                "result": "file_organize: 1 file processed, 1 moved, 0 errors.\n  ✓  CV_John_2025.pdf  →  ~/Documents/Job/",
            },
        ],
        "final_thought": "The resume has been moved to the Job folder.",
        "final_answer": "Done. I found CV_John_2025.pdf in ~/Downloads/ and moved it to ~/Documents/Job/. I also found a draft at ~/Desktop/resume_draft.docx — let me know if you want to move that too.",
        "language": "en",
        "category": "file_organize",
    },

    # --- ls para ver estructura antes de mover ---
    {
        "user_request": "Organiza las fotos de ~/Descargas por año en ~/Fotos/",
        "steps": [
            {
                "thought": "Primero listo Descargas para ver qué hay.",
                "tool": "process_run",
                "args": {"name": "ls", "args": ["-la", "~/Descargas"], "dry_run": False},
                "result": "process_run: 'ls -la ~/Descargas' → código 0\ntotal 24\ndrwxr-xr-x  img_2024_vacaciones.jpg\ndrwxr-xr-x  foto_2024_cumple.jpg\ndrwxr-xr-x  foto_2025_navidad.jpg\ndrwxr-xr-x  img_2025_verano.jpg\ndrwxr-xr-x  captura_2026_mayo.png",
            },
            {
                "thought": "Hay fotos de 2024, 2025 y 2026. Muevo las de 2024 primero.",
                "tool": "file_organize",
                "args": {"files": ["~/Descargas/img_2024_vacaciones.jpg", "~/Descargas/foto_2024_cumple.jpg"], "dest": "~/Fotos/2024/", "dry_run": False},
                "result": "file_organize: 2 archivos procesados, 2 a mover, 0 errores.\n  ✓  img_2024_vacaciones.jpg  →  ~/Fotos/2024/\n  ✓  foto_2024_cumple.jpg  →  ~/Fotos/2024/",
            },
            {
                "thought": "Ahora las de 2025.",
                "tool": "file_organize",
                "args": {"files": ["~/Descargas/foto_2025_navidad.jpg", "~/Descargas/img_2025_verano.jpg"], "dest": "~/Fotos/2025/", "dry_run": False},
                "result": "file_organize: 2 archivos procesados, 2 a mover, 0 errores.\n  ✓  foto_2025_navidad.jpg  →  ~/Fotos/2025/\n  ✓  img_2025_verano.jpg  →  ~/Fotos/2025/",
            },
            {
                "thought": "Y la de 2026.",
                "tool": "file_organize",
                "args": {"files": ["~/Descargas/captura_2026_mayo.png"], "dest": "~/Fotos/2026/", "dry_run": False},
                "result": "file_organize: 1 archivo procesado, 1 a mover, 0 errores.\n  ✓  captura_2026_mayo.png  →  ~/Fotos/2026/",
            },
        ],
        "final_thought": "Todas las fotos están ahora ordenadas por año.",
        "final_answer": "Fotos organizadas por año en ~/Fotos/: 2 fotos en 2024/, 2 en 2025/ y 1 en 2026/. Total: 5 archivos movidos.",
        "language": "es",
        "category": "file_organize",
    },

    # =========================================================================
    # file_organize — variedad adicional
    # =========================================================================

    # --- mover archivos de código a proyecto ---
    {
        "user_request": "Mueve todos los scripts Python del escritorio a ~/Proyectos/scripts/",
        "steps": [
            {
                "thought": "Busco todos los .py en el escritorio.",
                "tool": "search_files",
                "args": {"query": "", "path": "~/Desktop", "extensions": [".py"]},
                "result": "search_files: 4 coincidencia(s) en 4 archivo(s).\n  ~/Desktop/analisis.py\n  ~/Desktop/scraper.py\n  ~/Desktop/utils.py\n  ~/Desktop/test_api.py",
            },
            {
                "thought": "Muevo los 4 scripts a la carpeta de proyectos.",
                "tool": "file_organize",
                "args": {"files": ["~/Desktop/analisis.py", "~/Desktop/scraper.py", "~/Desktop/utils.py", "~/Desktop/test_api.py"], "dest": "~/Proyectos/scripts/", "dry_run": False},
                "result": "file_organize: 4 archivos procesados, 4 a mover, 0 errores.\n  ✓  analisis.py  →  ~/Proyectos/scripts/\n  ✓  scraper.py  →  ~/Proyectos/scripts/\n  ✓  utils.py  →  ~/Proyectos/scripts/\n  ✓  test_api.py  →  ~/Proyectos/scripts/",
            },
        ],
        "final_thought": "Los 4 scripts Python están ahora en ~/Proyectos/scripts/.",
        "final_answer": "Movidos 4 scripts Python del escritorio a ~/Proyectos/scripts/: analisis.py, scraper.py, utils.py y test_api.py.",
        "language": "es",
        "category": "file_organize",
    },

    # --- English: organize by type ---
    {
        "user_request": "Clean up Downloads: move videos to ~/Videos and archives to ~/Archives",
        "steps": [
            {
                "thought": "Search for video files in Downloads.",
                "tool": "search_files",
                "args": {"query": "", "path": "~/Downloads", "extensions": [".mp4", ".mkv", ".avi", ".mov"]},
                "result": "search_files: 2 match(es) in 2 file(s).\n  ~/Downloads/tutorial_python.mp4\n  ~/Downloads/movie_2026.mkv",
            },
            {
                "thought": "Move videos to ~/Videos.",
                "tool": "file_organize",
                "args": {"files": ["~/Downloads/tutorial_python.mp4", "~/Downloads/movie_2026.mkv"], "dest": "~/Videos/", "dry_run": False},
                "result": "file_organize: 2 files processed, 2 moved, 0 errors.\n  ✓  tutorial_python.mp4  →  ~/Videos/\n  ✓  movie_2026.mkv  →  ~/Videos/",
            },
            {
                "thought": "Now search for archives.",
                "tool": "search_files",
                "args": {"query": "", "path": "~/Downloads", "extensions": [".zip", ".tar", ".gz", ".rar"]},
                "result": "search_files: 3 match(es) in 3 file(s).\n  ~/Downloads/backup_2025.zip\n  ~/Downloads/project_src.tar.gz\n  ~/Downloads/fonts_pack.zip",
            },
            {
                "thought": "Move archives to ~/Archives.",
                "tool": "file_organize",
                "args": {"files": ["~/Downloads/backup_2025.zip", "~/Downloads/project_src.tar.gz", "~/Downloads/fonts_pack.zip"], "dest": "~/Archives/", "dry_run": False},
                "result": "file_organize: 3 files processed, 3 moved, 0 errors.\n  ✓  backup_2025.zip  →  ~/Archives/\n  ✓  project_src.tar.gz  →  ~/Archives/\n  ✓  fonts_pack.zip  →  ~/Archives/",
            },
        ],
        "final_thought": "Downloads is now clean: videos and archives organized.",
        "final_answer": "Downloads cleaned up: moved 2 videos to ~/Videos/ and 3 archives to ~/Archives/.",
        "language": "en",
        "category": "file_organize",
    },

    # =========================================================================
    # email_filter — variedad adicional
    # =========================================================================

    # --- filtro por dominio ---
    {
        "user_request": "Muéstrame todos los correos que llegaron de '@empresa.com' esta semana",
        "steps": [
            {
                "thought": "Filtro por remitente del dominio empresa.com en los últimos 7 días.",
                "tool": "email_filter",
                "args": {"sender": "@empresa.com", "date_range": "week"},
                "result": "email_filter: 3 correo(s).\n  De: ana@empresa.com | 13 may 2026 | 'Revisión de KPIs'\n  De: pedro@empresa.com | 14 may 2026 | 'Actualización del roadmap'\n  De: soporte@empresa.com | 15 may 2026 | 'Ticket #4521 resuelto'",
            },
        ],
        "final_thought": "Hay 3 correos de empresa.com esta semana.",
        "final_answer": "Esta semana tienes 3 correos de @empresa.com: 'Revisión de KPIs' (Ana, 13 mayo), 'Actualización del roadmap' (Pedro, 14 mayo) y 'Ticket #4521 resuelto' (Soporte, 15 mayo).",
        "language": "es",
        "category": "email_filter",
    },

    # --- sin resultados ---
    {
        "user_request": "¿He recibido algún correo de mi banco esta semana?",
        "steps": [
            {
                "thought": "Filtro correos del banco en los últimos 7 días.",
                "tool": "email_filter",
                "args": {"sender": "banco", "date_range": "week"},
                "result": "email_filter: 0 correo(s).",
            },
            {
                "thought": "No hay resultados esta semana. Pruebo con el mes completo.",
                "tool": "email_filter",
                "args": {"sender": "banco", "date_range": "month"},
                "result": "email_filter: 1 correo(s).\n  De: alertas@bancosantander.es | 05 may 2026 | 'Movimiento en tu cuenta: -120,00€'",
            },
        ],
        "final_thought": "No hay correos del banco esta semana, pero sí uno el 5 de mayo.",
        "final_answer": "Esta semana no has recibido correos del banco. El último fue el 5 de mayo: 'Movimiento en tu cuenta: -120,00€' de bancosantander.es.",
        "language": "es",
        "category": "email_filter",
    },

    # --- múltiples criterios ---
    {
        "user_request": "¿Tengo correos sin leer con asunto que mencione 'factura'?",
        "steps": [
            {
                "thought": "Filtro correos no leídos con 'factura' en el asunto.",
                "tool": "email_filter",
                "args": {"subject": "factura", "unread": True},
                "result": "email_filter: 2 correo(s).\n  De: contabilidad@proveedor.es | 12 may 2026 | 'Factura #2026-089 adjunta' [NO LEÍDO]\n  De: noreply@aws.amazon.com | 01 may 2026 | 'Tu factura de AWS — mayo 2026' [NO LEÍDO]",
            },
        ],
        "final_thought": "Hay 2 facturas sin leer.",
        "final_answer": "Tienes 2 facturas sin leer: la #2026-089 de tu proveedor (12 mayo) y la de AWS de mayo (1 mayo).",
        "language": "es",
        "category": "email_filter",
    },

    # --- English ---
    {
        "user_request": "Show me unread emails from GitHub",
        "steps": [
            {
                "thought": "Filter unread emails from GitHub.",
                "tool": "email_filter",
                "args": {"sender": "github.com", "unread": True},
                "result": "email_filter: 4 email(s).\n  From: noreply@github.com | 15 May 2026 | '[your-repo] PR #42 merged' [UNREAD]\n  From: noreply@github.com | 14 May 2026 | '[your-repo] New issue: memory leak' [UNREAD]\n  From: noreply@github.com | 13 May 2026 | '[your-repo] PR #41 review requested' [UNREAD]\n  From: noreply@github.com | 12 May 2026 | 'Security alert: dependency update needed' [UNREAD]",
            },
        ],
        "final_thought": "4 unread GitHub emails, including a security alert.",
        "final_answer": "You have 4 unread emails from GitHub: PR #42 merged, a new 'memory leak' issue, a review request for PR #41, and a security alert about a dependency update. I'd prioritize the security alert.",
        "language": "en",
        "category": "email_filter",
    },

    # =========================================================================
    # calendar_get — variedad adicional
    # =========================================================================

    # --- próximos 3 días ---
    {
        "user_request": "¿Qué tengo los próximos 3 días?",
        "steps": [
            {
                "thought": "Consulto los próximos 3 días del calendario.",
                "tool": "calendar_get",
                "args": {"date_range": "2026-05-15/2026-05-17"},
                "result": "calendar_get (2026-05-15/2026-05-17): 3 evento(s).\n  2026-05-15  Dentista 10:30  @Clínica Norte\n  2026-05-16  Sin eventos\n  2026-05-17  Revisión de sprint  @Sala A  14:00",
            },
        ],
        "final_thought": "Los próximos 3 días tienen 2 eventos reales.",
        "final_answer": "Próximos 3 días: mañana dentista a las 10:30 (Clínica Norte), el 16 libre, y el 17 revisión de sprint a las 14:00 en la Sala A.",
        "language": "es",
        "category": "calendar_get",
    },

    # --- evento por nombre ---
    {
        "user_request": "¿Cuándo es la próxima revisión trimestral?",
        "steps": [
            {
                "thought": "Busco eventos con 'trimestral' en los próximos 60 días.",
                "tool": "calendar_get",
                "args": {"date_range": "month", "subject": "trimestral"},
                "result": "calendar_get (month, subject=trimestral): 1 evento(s).\n  2026-05-28  Revisión trimestral  @Sala de conferencias  09:00-12:00",
            },
        ],
        "final_thought": "La próxima revisión trimestral es el 28 de mayo.",
        "final_answer": "La próxima revisión trimestral es el 28 de mayo de 09:00 a 12:00 en la Sala de conferencias.",
        "language": "es",
        "category": "calendar_get",
    },

    # --- English: this weekend ---
    {
        "user_request": "What do I have this weekend?",
        "steps": [
            {
                "thought": "Let me check Saturday and Sunday.",
                "tool": "calendar_get",
                "args": {"date_range": "2026-05-16/2026-05-17"},
                "result": "calendar_get (2026-05-16/2026-05-17): 1 event(s).\n  2026-05-17  Family lunch  @Parents' house  13:00",
            },
        ],
        "final_thought": "One event this weekend: family lunch on Sunday.",
        "final_answer": "This weekend you only have one event: family lunch at your parents' house on Sunday at 13:00.",
        "language": "en",
        "category": "calendar_get",
    },

    # =========================================================================
    # note_save — variedad adicional
    # =========================================================================

    # --- nota con contexto largo ---
    {
        "user_request": "Guarda el resultado del standup de hoy: Ayer: terminé el módulo de inferencia. Hoy: empiezo el dataset. Bloqueantes: ninguno.",
        "steps": [
            {
                "thought": "Guardo el standup con un título con la fecha de hoy.",
                "tool": "note_save",
                "args": {"title": "Standup 15 mayo 2026", "body": "Ayer: terminé el módulo de inferencia.\nHoy: empiezo el dataset.\nBloqueantes: ninguno."},
                "result": "note_save: nota guardada en '~/Notas/20260515_091500_Standup 15 mayo 2026.txt' (98 caracteres).",
            },
        ],
        "final_thought": "El standup ha quedado guardado.",
        "final_answer": "Standup del 15 mayo guardado en ~/Notas/.",
        "language": "es",
        "category": "note_save",
    },

    # --- English: quick capture ---
    {
        "user_request": "Save a quick note: 'Meeting with client moved to 4pm'",
        "steps": [
            {
                "thought": "Saving this quick note.",
                "tool": "note_save",
                "args": {"title": "Meeting reschedule", "body": "Meeting with client moved to 4pm — check calendar update."},
                "result": "note_save: note saved at '~/Notes/20260515_103422_Meeting reschedule.txt' (68 characters).",
            },
        ],
        "final_thought": "Note saved.",
        "final_answer": "Done. Quick note 'Meeting with client moved to 4pm' saved in ~/Notes/.",
        "language": "en",
        "category": "note_save",
    },

    # --- nota en carpeta específica ---
    {
        "user_request": "Guarda estas ideas de negocio en ~/Documentos/Ideas/: suscripción premium, integración con Slack, app móvil",
        "steps": [
            {
                "thought": "Guardo las ideas en la carpeta especificada.",
                "tool": "note_save",
                "args": {"title": "Ideas negocio mayo 2026", "body": "1. Suscripción premium\n2. Integración con Slack\n3. App móvil", "folder": "~/Documentos/Ideas/"},
                "result": "note_save: nota guardada en '~/Documentos/Ideas/20260515_Ideas negocio mayo 2026.txt' (78 caracteres).",
            },
        ],
        "final_thought": "Las ideas están guardadas en la carpeta correcta.",
        "final_answer": "Nota guardada en ~/Documentos/Ideas/ con las 3 ideas de negocio: suscripción premium, integración con Slack y app móvil.",
        "language": "es",
        "category": "note_save",
    },

    # --- nota + búsqueda para no duplicar ---
    {
        "user_request": "Añade 'mantequilla' a mi lista de la compra",
        "steps": [
            {
                "thought": "Busco si ya existe una nota de lista de la compra para no duplicar.",
                "tool": "search_files",
                "args": {"query": "lista de la compra", "path": "~/Notas", "extensions": [".txt", ".md"]},
                "result": "search_files: 1 coincidencia(s) en 1 archivo(s).\n  ~/Notas/20260514_170344_Lista de la compra.txt  (línea 1)\n    →  1: - Leche",
            },
            {
                "thought": "Ya existe la nota. La actualizo con el nuevo ítem.",
                "tool": "note_save",
                "args": {"title": "Lista de la compra", "body": "- Leche\n- Pan\n- Huevos\n- Aceite de oliva\n- Café\n- Mantequilla"},
                "result": "note_save: nota guardada en '~/Notas/20260515_090000_Lista de la compra.txt' (78 caracteres).",
            },
        ],
        "final_thought": "La lista de la compra ha sido actualizada con mantequilla.",
        "final_answer": "Añadido 'mantequilla' a la lista de la compra. La lista actualizada tiene: leche, pan, huevos, aceite de oliva, café y mantequilla.",
        "language": "es",
        "category": "note_save",
    },

    # =========================================================================
    # search_files — variedad adicional + discovery
    # =========================================================================

    # --- búsqueda en múltiples extensiones ---
    {
        "user_request": "Busca todos los archivos de configuración que tengan la clave 'API_KEY'",
        "steps": [
            {
                "thought": "Busco 'API_KEY' en archivos de configuración habituales.",
                "tool": "search_files",
                "args": {"query": "API_KEY", "path": "~", "extensions": [".env", ".yaml", ".yml", ".json", ".toml", ".ini"]},
                "result": "search_files: 3 coincidencia(s) en 2 archivo(s) (28 revisados).\n  ~/Proyecto_V3/.env  (línea 3)\n    →  3: API_KEY=sk-...\n  ~/Proyecto_V3/.env  (línea 7)\n    →  7: OPENAI_API_KEY=sk-...\n  ~/scripts/config.yaml  (línea 12)\n    →  12: api_key: sk-proj-abc123",
            },
        ],
        "final_thought": "La clave API_KEY aparece en .env (2 veces) y config.yaml.",
        "final_answer": "Encontré 'API_KEY' en 2 archivos: .env (2 veces: API_KEY y OPENAI_API_KEY) y scripts/config.yaml. Asegúrate de que .env no esté en el repositorio.",
        "language": "es",
        "category": "search_files",
    },

    # --- búsqueda con ruta desconocida (discovery) ---
    {
        "user_request": "No recuerdo dónde guardé el archivo de configuración de nginx",
        "steps": [
            {
                "thought": "Busco archivos llamados 'nginx' o con contenido relacionado.",
                "tool": "search_files",
                "args": {"query": "nginx", "path": "~", "extensions": [".conf", ".cfg", ".txt", ".yaml"]},
                "result": "search_files: 1 coincidencia(s) en 1 archivo(s) (34 revisados).\n  ~/servers/nginx.conf  (línea 1)\n    →  1: # Configuración nginx v1.24",
            },
        ],
        "final_thought": "El archivo nginx.conf está en ~/servers/.",
        "final_answer": "El archivo de configuración de nginx está en ~/servers/nginx.conf.",
        "language": "es",
        "category": "search_files",
    },

    # --- sin resultados + búsqueda ampliada ---
    {
        "user_request": "Busca el archivo 'presupuesto_definitivo.xlsx' en Documentos",
        "steps": [
            {
                "thought": "Busco el archivo exacto en Documentos.",
                "tool": "search_files",
                "args": {"query": "presupuesto_definitivo", "path": "~/Documentos", "extensions": [".xlsx"]},
                "result": "search_files: ninguna coincidencia para 'presupuesto_definitivo' en '~/Documentos' (12 archivos revisados).",
            },
            {
                "thought": "No está. Amplío la búsqueda a todo el home con nombre parcial.",
                "tool": "search_files",
                "args": {"query": "presupuesto", "path": "~", "extensions": [".xlsx", ".xls", ".ods"]},
                "result": "search_files: 2 coincidencia(s) en 2 archivo(s).\n  ~/Descargas/presupuesto_borrador.xlsx  (línea 1)\n  ~/Documentos/Trabajo/presupuesto_v2.xlsx  (línea 1)",
            },
        ],
        "final_thought": "El archivo 'presupuesto_definitivo' no existe, pero hay dos similares.",
        "final_answer": "No encontré 'presupuesto_definitivo.xlsx' en Documentos. Sí hay archivos similares: presupuesto_borrador.xlsx (Descargas) y presupuesto_v2.xlsx (Documentos/Trabajo). ¿Es alguno de estos?",
        "language": "es",
        "category": "search_files",
    },

    # --- English: code search ---
    {
        "user_request": "Find all files that use the deprecated 'loadModel' function",
        "steps": [
            {
                "thought": "Searching for 'loadModel' in Python and JavaScript files.",
                "tool": "search_files",
                "args": {"query": "loadModel", "path": "~/projects", "extensions": [".py", ".js", ".ts"]},
                "result": "search_files: 5 match(es) in 4 file(s) (67 reviewed).\n  ~/projects/api/handlers.py  (line 12)\n    →  12: model = loadModel('gpt4')\n  ~/projects/api/handlers.py  (line 45)\n    →  45: fallback = loadModel('gpt3')\n  ~/projects/ui/App.js  (line 8)\n    →  8: import { loadModel } from './utils'\n  ~/projects/ui/components/Chat.tsx  (line 23)\n    →  23: const m = loadModel(config.modelId)\n  ~/projects/tests/test_api.py  (line 15)\n    →  15: assert loadModel('test') is not None",
            },
        ],
        "final_thought": "Found 5 usages in 4 files.",
        "final_answer": "Found 'loadModel' in 4 files (5 usages): handlers.py (lines 12, 45), App.js (line 8), Chat.tsx (line 23), test_api.py (line 15). All need updating to the new API.",
        "language": "en",
        "category": "search_files",
    },

    # =========================================================================
    # read_file + análisis de código (T3 benchmark)
    # =========================================================================

    {
        "user_request": "Lee el archivo motor/agent.py y dime qué herramientas tiene disponibles el agente",
        "steps": [
            {
                "thought": "Leo el archivo agent.py para ver las herramientas.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/motor/agent.py"},
                "result": "Archivo leído: ~/Proyecto_V3/motor/agent.py\n\n# --- read_file ---\ndef _tool_read_file(path: str) -> str:\n    ...\n\n# --- list_dir ---\ndef _tool_list_dir(path: str = \".\") -> str:\n    ...\n\n# --- shell ---\ndef _tool_shell(command: str) -> str:\n    ...\n\n# --- http_get ---\ndef _tool_http_get(url: str) -> str:\n    ...",
            },
        ],
        "final_thought": "El archivo muestra 4 herramientas: read_file, list_dir, shell, http_get.",
        "final_answer": "El agente en motor/agent.py tiene 4 herramientas: read_file (lee archivos), list_dir (lista directorios), shell (ejecuta comandos de solo lectura) y http_get (peticiones HTTP GET).",
        "language": "es",
        "category": "read_file",
    },

    {
        "user_request": "Mira el archivo train_domestic.py y dime qué modelo base usa y cuántas épocas entrena",
        "steps": [
            {
                "thought": "Leo train_domestic.py para ver la configuración.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/train_domestic.py"},
                "result": "Archivo leído: ~/Proyecto_V3/train_domestic.py\n\nMODEL_ID = \"Qwen/Qwen2.5-3B-Instruct\"\nDATASET  = \"datasets/dataset_domestic_v4.jsonl\"\nEPOCHS = 3\nBATCH_SIZE = 4",
            },
        ],
        "final_thought": "El script usa Qwen2.5-3B-Instruct con 3 épocas.",
        "final_answer": "El archivo train_domestic.py usa el modelo base Qwen/Qwen2.5-3B-Instruct y entrena durante 3 épocas con batch size 4.",
        "language": "es",
        "category": "read_file",
    },

    {
        "user_request": "Revisa el README.md del proyecto y hazme un resumen de 3 puntos clave",
        "steps": [
            {
                "thought": "Leo el README.md para extraer los puntos principales.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/README.md"},
                "result": "Archivo leído: ~/Proyecto_V3/README.md\n\n# Motor de LoRAs Custom\n\nFábrica automatizada de adapters LoRA.\n- DataDigestor: convierte datos crudos en datasets de entrenamiento\n- ModelAnalyzer: analiza modelos y recomienda configuraciones\n- LLMTrainer: entrena adapters LoRA con PEFT+TRL\n- ExportManager: exporta a safetensors y GGUF",
            },
        ],
        "final_thought": "El README describe 3 componentes principales.",
        "final_answer": "Resumen del README: (1) DataDigestor convierte datos crudos a datasets, (2) ModelAnalyzer recomienda configuraciones óptimas de LoRA, (3) LLMTrainer entrena adapters que se pueden exportar a GGUF para usar sin GPU.",
        "language": "es",
        "category": "read_file",
    },

    {
        "user_request": "Busca el archivo de configuración del adapter domestic_base y dime qué rank de LoRA usa",
        "steps": [
            {
                "thought": "Primero busco el archivo adapter_config.json del adapter domestic_base.",
                "tool": "search_files",
                "args": {"query": "domestic_base", "path": "~/Proyecto_V3/adapters", "extensions": [".json"]},
                "result": "search_files: 2 coincidencia(s) en 2 archivo(s).\n  ~/Proyecto_V3/adapters/domestic_base/adapter_config.json\n  ~/Proyecto_V3/adapters/domestic_base/meta.json",
            },
            {
                "thought": "Leo el adapter_config.json para ver el rank de LoRA.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/adapters/domestic_base/adapter_config.json"},
                "result": "Archivo leído: ~/Proyecto_V3/adapters/domestic_base/adapter_config.json\n\n{\n  \"r\": 16,\n  \"lora_alpha\": 32,\n  \"target_modules\": [\"q_proj\", \"v_proj\", \"k_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n  \"base_model_name_or_path\": \"Qwen/Qwen2.5-3B-Instruct\"\n}",
            },
        ],
        "final_thought": "El adapter domestic_base usa r=16 con alpha=32.",
        "final_answer": "El adapter domestic_base usa LoRA rank r=16, alpha=32, sobre Qwen2.5-3B-Instruct, con 7 módulos objetivo (todos los de atención y MLP).",
        "language": "es",
        "category": "read_file",
    },

    # =========================================================================
    # read_file + list_dir encadenados (navegación de directorios)
    # =========================================================================

    {
        "user_request": "Explora la carpeta adapters/ y dime qué adapters hay entrenados y cuál tiene el mejor rendimiento",
        "steps": [
            {
                "thought": "Primero listo el contenido de adapters/ para ver qué carpetas hay.",
                "tool": "list_dir",
                "args": {"path": "~/Proyecto_V3/adapters"},
                "result": "~/Proyecto_V3/adapters/\n  domestic_base/\n  domestic_v2/\n  finance_sentiment_14b/",
            },
            {
                "thought": "Hay 3 adapters. Voy a leer el meta.json de cada uno para comparar rendimiento.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/adapters/domestic_base/meta.json"},
                "result": "Archivo leído: meta.json\n{\"eval_loss\": 0.1, \"train_loss\": 0.37, \"elapsed_min\": 47.5}",
            },
            {
                "thought": "Ahora leo el meta del domestic_v2.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/adapters/domestic_v2/meta.json"},
                "result": "Archivo leído: meta.json\n{\"eval_loss\": 0.0009, \"train_loss\": 0.58, \"elapsed_min\": 22.3}",
            },
        ],
        "final_thought": "Comparando eval_loss: domestic_v2 (0.0009) es mejor que domestic_base (0.1).",
        "final_answer": "Hay 3 adapters en adapters/: domestic_base (eval_loss=0.1), domestic_v2 (eval_loss=0.0009) y finance_sentiment_14b. El de mejor rendimiento es domestic_v2 con eval_loss de 0.0009.",
        "language": "es",
        "category": "read_file",
    },

    {
        "user_request": "¿Cuántos archivos Python hay en el directorio motor/ y cuál es el más grande?",
        "steps": [
            {
                "thought": "Listo el directorio motor/ para ver los archivos.",
                "tool": "list_dir",
                "args": {"path": "~/Proyecto_V3/motor"},
                "result": "~/Proyecto_V3/motor/\n  __init__.py  (1,234 B)\n  agent.py  (22,456 B)\n  analyzer.py  (18,234 B)\n  continual.py  (31,567 B)\n  digestor.py  (98,234 B)\n  domestic_dataset_gen.py  (67,890 B)\n  domestic_tools.py  (15,432 B)\n  exporter.py  (12,345 B)\n  report.py  (8,765 B)\n  server.py  (25,678 B)\n  trainer_llm.py  (28,901 B)\n  trainer_vlm.py  (9,876 B)",
            },
        ],
        "final_thought": "Conté 9 archivos .py. El más grande es digestor.py con 98 KB.",
        "final_answer": "En motor/ hay 9 archivos Python. El más grande es digestor.py (98 KB), seguido de domestic_dataset_gen.py (68 KB) y continual.py (32 KB).",
        "language": "es",
        "category": "list_dir",
    },

    # =========================================================================
    # Manejo de errores y reintentos (robustez)
    # =========================================================================

    {
        "user_request": "Lee el archivo config.yaml que debería estar en ~/Proyecto_V3/",
        "steps": [
            {
                "thought": "Intento leer config.yaml directamente.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/config.yaml"},
                "result": "Archivo no encontrado: ~/Proyecto_V3/config.yaml",
            },
            {
                "thought": "No existe config.yaml. Busco archivos YAML en el proyecto por si tiene otro nombre.",
                "tool": "search_files",
                "args": {"query": "", "path": "~/Proyecto_V3", "extensions": [".yaml", ".yml"]},
                "result": "search_files: 0 coincidencia(s) en 0 archivo(s) (5 revisados).\n  No se encontraron archivos con extensiones: .yaml, .yml",
            },
        ],
        "final_thought": "No existe ningún archivo YAML en el proyecto.",
        "final_answer": "No encontré config.yaml ni ningún otro archivo YAML en ~/Proyecto_V3/. ¿Puedes verificar la ubicación o el nombre del archivo?",
        "language": "es",
        "category": "read_file",
    },

    {
        "user_request": "Busca el archivo training_log.txt en ~/Documentos/",
        "steps": [
            {
                "thought": "Busco training_log.txt en ~/Documentos/.",
                "tool": "search_files",
                "args": {"query": "training_log", "path": "~/Documentos", "extensions": [".txt", ".log"]},
                "result": "search_files: ninguna coincidencia para 'training_log' en '~/Documentos' (15 archivos revisados).",
            },
            {
                "thought": "No está en Documentos. Amplío la búsqueda a todo el home.",
                "tool": "search_files",
                "args": {"query": "loss", "path": "~", "extensions": [".txt", ".log"]},
                "result": "search_files: 2 coincidencia(s) en 2 archivo(s).\n  ~/Proyecto_V3/train_log.txt  (línea 1)\n  ~/Descargas/experiment_log.txt  (línea 5)",
            },
        ],
        "final_thought": "No existe 'training_log.txt', pero encontré archivos de log relacionados.",
        "final_answer": "No encontré 'training_log.txt' en ~/Documentos/. Sin embargo, hay archivos de log en ~/Proyecto_V3/train_log.txt y ~/Descargas/experiment_log.txt que podrían contener la información que buscas.",
        "language": "es",
        "category": "search_files",
    },

    # =========================================================================
    # Síntesis multi-documento (T4 benchmark)
    # =========================================================================

    {
        "user_request": "Compara los adapters domestic_base y domestic_v2: ¿cuál entrena más rápido y con mejor loss?",
        "steps": [
            {
                "thought": "Leo los meta.json de ambos adapters para comparar.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/adapters/domestic_base/meta.json"},
                "result": "Archivo leído: meta.json\n{\"eval_loss\": 0.1, \"train_loss\": 0.37, \"elapsed_min\": 47.5, \"train_samples\": 630}",
            },
            {
                "thought": "Ahora el meta de domestic_v2.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/adapters/domestic_v2/meta.json"},
                "result": "Archivo leído: meta.json\n{\"eval_loss\": 0.0009, \"train_loss\": 0.58, \"elapsed_min\": 22.3, \"train_samples\": 240}",
            },
        ],
        "final_thought": "domestic_v2 es más rápido y tiene mejor eval_loss que domestic_base.",
        "final_answer": "Comparativa:\n- domestic_base: 47.5 min, eval_loss=0.1, 630 muestras\n- domestic_v2: 22.3 min, eval_loss=0.0009, 240 muestras\nEl domestic_v2 entrena 2.1× más rápido y tiene eval_loss 111× mejor, aunque usa menos datos (dataset más limpio).",
        "language": "es",
        "category": "read_file",
    },

    {
        "user_request": "Analiza los tests del proyecto: ¿cuántos hay y qué cubren?",
        "steps": [
            {
                "thought": "Listo la carpeta de tests para ver qué archivos hay.",
                "tool": "list_dir",
                "args": {"path": "~/Proyecto_V3/tests"},
                "result": "~/Proyecto_V3/tests/\n  test_agent_robustness.py  (45,234 B)\n  test_continual_e2e.py  (38,901 B)\n  test_domestic.py  (22,345 B)\n  test_g2.py  (15,678 B)\n  test_session_15mayo.py  (28,456 B)",
            },
            {
                "thought": "Leo el test principal para ver cobertura.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/tests/test_agent_robustness.py"},
                "result": "Archivo leído (primeras 40 líneas): test_agent_robustness.py\n\nClases: TestSanitizeOutput, TestSanitizeObservation, TestExtractJsonObject, TestParseOutput, TestToolAliases, TestSystemPrompt, TestAgentRunWithMock, TestTools, TestAgentResult, TestToolWrapper, TestDefaultTools, TestShellBlocked\n\n98 tests en total cubriendo: sanitización, parsing JSON, aliases de herramientas, system prompt, mock de inferencia, seguridad shell.",
            },
        ],
        "final_thought": "El proyecto tiene 5 archivos de tests con 98+ tests.",
        "final_answer": "El proyecto tiene 5 archivos de tests en tests/: test_agent_robustness.py (98 tests de agente), test_continual_e2e.py (36 tests de continual learning), test_domestic.py (55 tests de herramientas), test_g2.py (30 tests de exportación) y test_session_15mayo.py (53 tests varios). Total: ~272 tests cubriendo agente, continual learning, herramientas domésticas, exportación y regresión.",
        "language": "es",
        "category": "read_file",
    },

    # =========================================================================
    # Tareas multi-paso complejas (T5 benchmark)
    # =========================================================================

    {
        "user_request": "Necesito saber si el proyecto está listo para demo: revisa los tests, el README, y dime si hay errores",
        "steps": [
            {
                "thought": "Primero reviso el README para ver el estado del proyecto.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/README.md"},
                "result": "Archivo leído: README.md\n\n✅ DataDigestor v3: 14 formatos\n✅ ModelAnalyzer v2: GPU profiles + recomendaciones\n✅ LLMTrainer: validado con Titanic, Finance, Emotion, Domestic\n✅ 285 tests en 3 capas\n⏳ VLMTrainer: pendiente test real",
            },
            {
                "thought": "Reviso si hay errores en los tests.",
                "tool": "shell",
                "args": {"command": "cd ~/Proyecto_V3 && python -m pytest tests/test_agent_robustness.py tests/test_domestic.py -q --tb=no 2>&1 | tail -5"},
                "result": "97 passed, 1 skipped in 0.20s\n55 passed in 0.15s",
            },
        ],
        "final_thought": "152 tests pasan. El README muestra todos los componentes en verde.",
        "final_answer": "El proyecto está listo para demo. Los tests pasan (152/153), todos los componentes principales están completos y validados. El único pendiente es VLMTrainer que no afecta al agente doméstico.",
        "language": "es",
        "category": "multi_step",
    },

    {
        "user_request": "Haz una auditoría rápida: mira cuánta VRAM hay, cuántos adapters hay, y si hay errores recientes en los logs",
        "steps": [
            {
                "thought": "Verifico la VRAM disponible.",
                "tool": "shell",
                "args": {"command": "nvidia-smi --query-gpu=memory.free --format=csv,noheader"},
                "result": "12288 MB",
            },
            {
                "thought": "Listo los adapters disponibles.",
                "tool": "list_dir",
                "args": {"path": "~/Proyecto_V3/adapters"},
                "result": "~/Proyecto_V3/adapters/\n  domestic_base/\n  domestic_v2/\n  domestic_7b/\n  finance_sentiment_14b/",
            },
            {
                "thought": "Busco logs de error recientes.",
                "tool": "search_files",
                "args": {"query": "ERROR", "path": "~/Proyecto_V3", "extensions": [".log", ".txt"]},
                "result": "search_files: 0 coincidencia(s) para 'ERROR' en archivos .log/.txt.",
            },
        ],
        "final_thought": "VRAM: 12 GB libres, 4 adapters, 0 errores en logs.",
        "final_answer": "Auditoría rápida:\n- VRAM libre: 12 GB\n- Adapters disponibles: 4 (domestic_base, domestic_v2, domestic_7b, finance_sentiment_14b)\n- Errores en logs: 0\nTodo en orden.",
        "language": "es",
        "category": "multi_step",
    },

    # =========================================================================
    # English: read_file + analysis
    # =========================================================================

    {
        "user_request": "Read the pyproject.toml and tell me what dependencies this project has",
        "steps": [
            {
                "thought": "Reading pyproject.toml to extract dependencies.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/pyproject.toml"},
                "result": "File read: pyproject.toml\n\n[build-system]\nrequires = [\"setuptools>=61.0\"]\n\n[project]\nname = \"motor-de-loras\"\ndependencies = [\"torch\", \"transformers\", \"peft\", \"trl\", \"bitsandbytes\", \"datasets\", \"accelerate\", \"scikit-learn\"]",
            },
        ],
        "final_thought": "The project depends on PyTorch, Transformers, PEFT, TRL, bitsandbytes, datasets, accelerate and scikit-learn.",
        "final_answer": "The project dependencies are: torch, transformers, peft, trl, bitsandbytes, datasets, accelerate, and scikit-learn. Build system uses setuptools.",
        "language": "en",
        "category": "read_file",
    },

    {
        "user_request": "Check the benchmark_report.html and tell me which model performed best",
        "steps": [
            {
                "thought": "Reading the benchmark report to extract results.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/benchmark_report.html"},
                "result": "File read (truncated): benchmark_report.html\n\n...3B+LoRA-v1: 4/5 tasks...\n...3B+LoRA-v2: 3/5 tasks...\n...7B+LoRA: 3/5 tasks...\n...T3 (code analysis): only 7B passed...",
            },
        ],
        "final_thought": "7B passed T3, both 3B models failed it. Overall success rates similar but 7B handles harder tasks.",
        "final_answer": "The benchmark shows 3B-v1 at 4/5, 3B-v2 at 3/5, and 7B at 3/5. However, 7B was the only model that passed T3 (code analysis) — the hardest task. 3B models are faster (31-157s) but struggle with deep analysis.",
        "language": "en",
        "category": "read_file",
    },

    # =========================================================================
    # English: multi-step
    # =========================================================================

    {
        "user_request": "Find all Markdown files that mention 'sprint' and summarize what sprints are completed",
        "steps": [
            {
                "thought": "Search for 'sprint' in Markdown files.",
                "tool": "search_files",
                "args": {"query": "sprint", "path": "~/Proyecto_V3", "extensions": [".md"], "max_results": 5},
                "result": "search_files: 5+ match(es) in 2 file(s) (4 reviewed).\n  ~/Proyecto_V3/README.md  (line 480)\n    →  480: ✅ Sprint 1-10 completados\n  ~/Proyecto_V3/SESION.md  (line 12)\n    →  12: Sprints 1-10 completados",
            },
            {
                "thought": "Reading README for sprint status.",
                "tool": "read_file",
                "args": {"path": "~/Proyecto_V3/README.md"},
                "result": "File read (truncated): README.md\n\nSprint 1-10 completed. Only S7 (Abliteration) postponed. CLI with 11 subcommands. 285 tests.",
            },
        ],
        "final_thought": "10 sprints completed, only S7 postponed.",
        "final_answer": "Found sprint info in README.md and SESION.md. Sprints 1-10 are completed (285 tests, 11 CLI subcommands). Only Sprint 7 (Abliteration) was postponed.",
        "language": "en",
        "category": "multi_step",
    },
]



# ---------------------------------------------------------------------------
# Aumentación ligera de los user_requests
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, list[str]] = {
    # Español — verbos de acción
    "organiza":   ["clasifica", "ordena", "agrupa"],
    "mueve":      ["traslada", "pasa", "coloca"],
    "busca":      ["encuentra", "localiza", "rastrea"],
    "guarda":     ["almacena", "salva", "graba"],
    "muéstrame":  ["enséñame", "lista", "dime"],
    "lista":      ["muestra", "enumera", "despliega"],
    "filtra":     ["selecciona", "extrae", "obtén"],
    "comprueba":  ["verifica", "revisa", "chequea"],
    "crea":       ["genera", "prepara", "redacta"],
    "añade":      ["agrega", "incluye", "incorpora"],
    "limpia":     ["despeja", "vacía", "reorganiza"],
    "abre":       ["lanza", "inicia", "ejecuta"],
    "dime":       ["cuéntame", "explícame", "muéstrame"],
    "tengo":      ["hay", "existen", "aparecen"],
    # Español — sustantivos
    "correos":    ["emails", "mensajes", "mails"],
    "archivos":   ["ficheros", "documentos", "archivos"],
    "carpeta":    ["directorio", "folder", "directorio"],
    "notas":      ["apuntes", "notas", "recordatorios"],
    "fotos":      ["imágenes", "fotografías", "capturas"],
    "proyectos":  ["repositorios", "carpetas", "directorios"],
    "escritorio": ["desktop", "escritorio", "pantalla"],
    "descargas":  ["downloads", "descargas", "bajadas"],
    "documentos": ["archivos", "ficheros", "documentos"],
    # English — verbs
    "find":       ["locate", "search for", "look for"],
    "move":       ["transfer", "copy", "relocate"],
    "show":       ["display", "list", "get"],
    "save":       ["store", "write", "keep"],
    "clean":      ["organize", "sort", "clear"],
    "check":      ["verify", "look at", "review"],
    "get":        ["fetch", "retrieve", "obtain"],
    # English — nouns
    "files":      ["documents", "items", "records"],
    "folder":     ["directory", "path", "location"],
    "emails":     ["messages", "mail", "inbox"],
    "notes":      ["reminders", "memos", "records"],
    "downloads":  ["downloads", "inbox", "received"],
}


def _augment_request(request: str, n: int = 3, seed: int = 0) -> list[str]:
    """
    Genera `n` variaciones del user_request sustituyendo sinónimos.
    Variación simple (no deep-learning) para diversificar el vocabulario.
    """
    rng = random.Random(seed)
    variants: list[str] = [request]
    for _ in range(n - 1):
        text = request
        words = text.split()
        for i, w in enumerate(words):
            key = w.lower().rstrip(".,;:?!")
            if key in _SYNONYMS:
                replacement = rng.choice(_SYNONYMS[key])
                # preserva capitalización y puntuación original
                suffix = w[len(key):]
                if w[0].isupper():
                    replacement = replacement.capitalize()
                words[i] = replacement + suffix
        variants.append(" ".join(words))
    return list(dict.fromkeys(variants))  # dedup manteniendo orden


# ---------------------------------------------------------------------------
# Adaptar ejemplos de Salesforce/xlam-function-calling-60k
# ---------------------------------------------------------------------------

# Mapeo de nombres de herramientas xlam → domestic_tools
_XLAM_TOOL_MAP: dict[str, str] = {
    "file_manager":         "file_organize",
    "move_file":            "file_organize",
    "organize_files":       "file_organize",
    "email_search":         "email_filter",
    "get_emails":           "email_filter",
    "filter_emails":        "email_filter",
    "calendar_events":      "calendar_get",
    "get_calendar":         "calendar_get",
    "list_events":          "calendar_get",
    "save_note":            "note_save",
    "create_note":          "note_save",
    "write_note":           "note_save",
    "file_search":          "search_files",
    "search_local_files":   "search_files",
    "find_in_files":        "search_files",
    "run_process":          "process_run",
    "execute_command":      "process_run",
}


def adapt_xlam(
    xlam_path: str,
    n:         int = 600,
    seed:      int = 42,
) -> list[DomesticExample]:
    """
    Filtra y adapta ejemplos de ``Salesforce/xlam-function-calling-60k`` al
    formato DomesticExample.

    El dataset xlam debe estar descargado como JSONL local. Estructura esperada
    de cada línea:
        {
          "query": "...",
          "answers": [{"name": "tool_name", "arguments": {...}}],
          ...
        }

    Solo se incluyen ejemplos que usen al menos una herramienta del mapeo
    _XLAM_TOOL_MAP. Los nombres de herramientas y parámetros se normalizan.

    Parámetros
    ----------
    xlam_path : ruta al JSONL de xlam
    n         : máximo de ejemplos a devolver (default: 600)
    seed      : semilla para el shuffle (default: 42)

    Devuelve
    --------
    Lista de DomesticExample en representación interna.
    """
    path = Path(xlam_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"xlam no encontrado: {xlam_path}")

    raw: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rng = random.Random(seed)
    rng.shuffle(raw)

    result: list[DomesticExample] = []
    for item in raw:
        if len(result) >= n:
            break

        query   = item.get("query", "").strip()
        answers = item.get("answers", [])
        if not query or not answers:
            continue

        # Filtrar: al menos un tool del mapeo
        matched_steps = []
        for ans in answers:
            tool_original = ans.get("name", "")
            mapped_tool   = _XLAM_TOOL_MAP.get(tool_original)
            if mapped_tool is None:
                continue
            args = ans.get("arguments", {})
            # Normalizar parámetros comunes
            args = _normalize_xlam_args(mapped_tool, args)
            matched_steps.append({
                "thought": f"Necesito usar {mapped_tool} para esta parte de la tarea.",
                "tool":    mapped_tool,
                "args":    args,
                "result":  f"[resultado simulado de {mapped_tool}]",
            })

        if not matched_steps:
            continue

        result.append({
            "user_request":  query,
            "steps":         matched_steps,
            "final_thought": "He completado la tarea.",
            "final_answer":  f"Tarea completada usando {', '.join(s['tool'] for s in matched_steps)}.",
            "language":      "en",
            "category":      matched_steps[0]["tool"],
        })

    return result


def _normalize_xlam_args(tool: str, args: dict) -> dict:
    """Mapea nombres de parámetros xlam a los nombres de nuestras firmas."""
    if tool == "file_organize":
        normalized = {}
        if "source" in args or "file" in args or "path" in args:
            src = args.get("source") or args.get("file") or args.get("path", "")
            normalized["files"] = [src] if isinstance(src, str) else src
        if "destination" in args or "dest" in args or "target" in args:
            normalized["dest"] = args.get("destination") or args.get("dest") or args.get("target", "~/Documents/")
        normalized.setdefault("files", [args.get("files", [])])
        normalized.setdefault("dest", "~/Documents/")
        normalized.setdefault("dry_run", True)
        return normalized

    if tool == "email_filter":
        normalized = {
            "server":   args.get("server", "imap.gmail.com"),
            "user":     args.get("email") or args.get("user", "user@example.com"),
            "password": "app-pass-1234",
            "folder":   args.get("folder", "INBOX"),
            "mode":     args.get("action") or args.get("mode", "list"),
            "dry_run":  True,
        }
        if "from" in args or "sender" in args:
            normalized["sender"] = args.get("from") or args.get("sender")
        if "subject" in args:
            normalized["subject"] = args["subject"]
        return normalized

    if tool == "calendar_get":
        date_range = (
            args.get("date_range")
            or args.get("date")
            or args.get("start_date", "today")
        )
        return {"date_range": date_range}

    if tool == "note_save":
        return {
            "title":  args.get("title", "Nota"),
            "body":   args.get("body") or args.get("content") or args.get("text", ""),
            "folder": args.get("folder", "~/Notas"),
        }

    if tool == "search_files":
        return {
            "query":      args.get("query") or args.get("keyword", ""),
            "path":       args.get("path") or args.get("directory", "~/"),
            "extensions": args.get("extensions"),
        }

    if tool == "process_run":
        return {
            "name":    args.get("name") or args.get("command", ""),
            "args":    args.get("args") or args.get("arguments", []),
            "dry_run": True,
        }

    return args


# ---------------------------------------------------------------------------
# Adaptar ejemplos de Team-ACE/ToolACE (dataset público, sin gating)
# ---------------------------------------------------------------------------

def _parse_toolace_call(text: str) -> list[dict]:
    """
    Parsea llamadas en formato ToolACE:  [FunctionName(arg1="v1", arg2=123)]
    Devuelve lista de {name, arguments}.
    """
    import ast as _ast
    calls = []
    # Nombre puede tener espacios: [ Market Trends API(args) ]
    for m in re.finditer(r"\[([^\[\](]+)\(([^)]*)\)\]", text):
        raw_name = m.group(1).strip()
        name = re.sub(r"\s+", "_", raw_name)   # spaces → underscores
        args_str = m.group(2).strip()
        args: dict = {}
        for kv in re.finditer(
            r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|-?\d+(?:\.\d+)?|\w+)',
            args_str,
        ):
            k = kv.group(1)
            v_str = kv.group(2)
            try:
                args[k] = _ast.literal_eval(v_str)
            except Exception:
                args[k] = v_str.strip("\"'")
        calls.append({"name": name, "arguments": args})
    return calls


def adapt_toolace(
    n:      int  = 600,
    seed:   int  = 42,
    min_steps: int = 1,
    max_steps: int = 3,
) -> list[DomesticExample]:
    """
    Descarga y adapta ejemplos de ``Team-ACE/ToolACE`` al formato DomesticExample.

    ToolACE contiene razonamiento general con herramientas (APIs de mercado,
    clima, calendarios, búsqueda, etc.). Estos ejemplos no usan nuestras firmas
    de domestic_tools pero enseñan al modelo el metapatrón de tool-calling:
    cuándo llamar una herramienta, cómo encadenar pasos, cómo usar el resultado.

    Las herramientas se normalizan (espacios → guiones bajos) y se representan
    en formato ReAct idéntico al que usa motor.agent.

    Parámetros
    ----------
    n         : máximo de ejemplos a devolver (default: 600)
    seed      : semilla para reproducibilidad (default: 42)
    min_steps : mínimo de pasos tool-call por ejemplo (default: 1)
    max_steps : máximo de pasos tool-call (se descartan los más largos)

    Requiere
    --------
    pip install datasets  (ya incluido en el entorno)
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Instala 'datasets': pip install datasets")

    ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)
    rng = random.Random(seed)
    result: list[DomesticExample] = []

    for item in ds:
        if len(result) >= n:
            break

        system_text = item.get("system", "")
        convs = item.get("conversations", [])
        if not convs:
            continue

        # Extraer user_request (primer turno de usuario)
        user_msg = next((c["value"] for c in convs if c.get("from") == "user"), "")
        if not user_msg:
            continue

        # Reconstruir pasos: pares (assistant_call, tool_result)
        steps: list[Step] = []
        final_answer = ""
        i = 0
        while i < len(convs):
            turn = convs[i]
            role  = turn.get("from", "")
            value = turn.get("value", "")

            if role == "assistant":
                calls = _parse_toolace_call(value)
                if calls:
                    # Hay tool calls — siguiente turno debería ser tool
                    tool_result = ""
                    if i + 1 < len(convs) and convs[i + 1].get("from") == "tool":
                        tool_result = convs[i + 1]["value"][:400]  # limitar longitud
                        i += 1  # saltar el turno tool
                    for call in calls:
                        thought = (
                            re.sub(r"\[.*", "", value).strip()
                            or f"Necesito obtener información usando {call['name']}."
                        )
                        steps.append({
                            "thought": thought[:200],
                            "tool":    call["name"],
                            "args":    call["arguments"],
                            "result":  tool_result or f"[resultado de {call['name']}]",
                        })
                else:
                    # Sin tool call → es la respuesta final
                    final_answer = value.strip()
            i += 1

        # Filtrar por número de pasos
        if not steps or not (min_steps <= len(steps) <= max_steps):
            continue
        if not final_answer:
            final_answer = "Task completed successfully."
        # Descartar ejemplos cuyo final_answer ES otra tool call (conversación incompleta).
        # Patrón de tool call: [Word(  o  [Word Word(  — sin ] antes del (
        # Esto NO captura Markdown links que tienen el patrón [text](url)
        if re.search(r"\[[\w\s]+\(", final_answer):
            continue

        result.append({
            "user_request":  user_msg[:300],
            "steps":         steps[:max_steps],
            "final_thought": "Ya tengo toda la información necesaria para responder.",
            "final_answer":  final_answer[:500],
            "language":      "en",
            "category":      steps[0]["tool"],
        })

    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# generate_dataset — punto de entrada principal
# ---------------------------------------------------------------------------

def generate_dataset(
    n:              int  = 2000,
    target_model:   str  = "react",
    out:            str  = "dataset_domestic.jsonl",
    xlam_path:      Optional[str] = None,
    xlam_n:         int  = 600,
    use_toolace:    bool = False,
    toolace_n:      int  = 600,
    augment_factor: int  = 3,
    seed:           int  = 42,
    verbose:        bool = True,
) -> list[dict]:
    """
    Genera el dataset base del agente doméstico y lo escribe en JSONL.

    Estrategia:
      70% propios: templates × augment_factor (variaciones de user_request)
      30% externos: adapt_toolace() (HF, sin gating) o adapt_xlam() (requiere descarga)

    Parámetros
    ----------
    n              : número total de ejemplos objetivo (default: 2000)
    target_model   : "react" | "qwen_native" (renderer)
    out            : ruta del JSONL de salida
    xlam_path      : ruta al JSONL de xlam (opcional); si None y use_toolace=False, solo templates
    xlam_n         : máximo de ejemplos xlam a incluir (default: 600)
    use_toolace    : si True, descarga y adapta ejemplos de Team-ACE/ToolACE (default: False)
    toolace_n      : máximo de ejemplos ToolACE a incluir (default: 600)
    augment_factor : variaciones de user_request por template base (default: 3)
    seed           : semilla para reproducibilidad (default: 42)
    verbose        : imprimir progreso (default: True)

    Devuelve
    --------
    Lista de dicts ChatML exportados (también escrita en `out`).
    """
    rng = random.Random(seed)

    # 1. Expandir templates con variaciones de user_request
    expanded: list[DomesticExample] = []
    for i, tmpl in enumerate(_TEMPLATES):
        variants = _augment_request(tmpl["user_request"], n=augment_factor, seed=seed + i)
        for v in variants:
            ex = dict(tmpl)
            ex["user_request"] = v
            expanded.append(ex)  # type: ignore[arg-type]

    if verbose:
        print(f"[domestic_dataset_gen] Templates expandidos: {len(expanded)} ejemplos")

    # 2. Añadir ejemplos externos (ToolACE o xlam)
    external_examples: list[DomesticExample] = []

    if use_toolace:
        if verbose:
            print(f"[domestic_dataset_gen] Descargando {toolace_n} ejemplos de ToolACE...")
        try:
            external_examples = adapt_toolace(n=toolace_n, seed=seed)
            if verbose:
                print(f"[domestic_dataset_gen] ToolACE adaptados: {len(external_examples)} ejemplos")
        except Exception as e:
            print(f"[domestic_dataset_gen] Error con ToolACE: {e}. Continuando solo con templates.")

    elif xlam_path:
        try:
            external_examples = adapt_xlam(xlam_path, n=xlam_n, seed=seed)
            if verbose:
                print(f"[domestic_dataset_gen] xlam adaptados: {len(external_examples)} ejemplos")
        except FileNotFoundError as e:
            print(f"[domestic_dataset_gen] Aviso: {e}. Continuando solo con templates.")

    # 3. Mezclar y limitar
    all_examples = expanded + external_examples
    rng.shuffle(all_examples)
    all_examples = all_examples[:n]

    if verbose:
        cats: dict[str, int] = {}
        for ex in all_examples:
            cats[ex["category"]] = cats.get(ex["category"], 0) + 1
        print(f"[domestic_dataset_gen] Total: {len(all_examples)} ejemplos")
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count} ({count/len(all_examples)*100:.1f}%)")

    # 4. Renderizar a ChatML
    dataset: list[dict] = []
    for ex in all_examples:
        try:
            dataset.append(to_chatml(ex, renderer=target_model))
        except Exception as e:
            if verbose:
                print(f"[domestic_dataset_gen] Error renderizando ejemplo: {e}")
            continue

    # 5. Escribir JSONL
    out_path = Path(out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if verbose:
        print(f"[domestic_dataset_gen] Dataset guardado en: {out_path}")

    return dataset
