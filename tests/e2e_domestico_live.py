# -*- coding: utf-8 -*-
"""
tests/e2e_domestico_live.py
===========================
Prueba E2E EN VIVO del loop agéntico doméstico (no se ejecuta con pytest:
requiere el servidor GGUF corriendo).

Flujo completo validado:
  1. Usuario pide en español mover archivos (sandbox temporal real)
  2. Gemma 4 elige la tool correcta entre dos disponibles y emite tool_call
  3. El motor ejecuta file_organize() DE VERDAD sobre el sandbox
  4. El resultado vuelve al modelo (role=tool) y este resume en español
  5. Se verifica en disco que los archivos se movieron

Uso:
    python tests/e2e_domestico_live.py [URL]     (default http://localhost:8002)
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motor.domestic_tools import file_organize  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_organize",
            "description": "Mueve una lista de archivos a una carpeta destino",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"},
                              "description": "Rutas de los archivos a mover"},
                    "dest":  {"type": "string", "description": "Carpeta destino"},
                },
                "required": ["files", "dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_save",
            "description": "Guarda una nota de texto",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body":  {"type": "string"},
                },
                "required": ["title", "body"],
            },
        },
    },
]


def chat(messages, tools=None):
    payload = {"messages": messages, "max_tokens": 300}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main() -> int:
    # Sandbox real con archivos desordenados — DENTRO de las rutas seguras
    # de domestic_tools (subcarpetas estándar del home; %TEMP% está
    # bloqueado por _is_safe_path, como comprobamos en la primera ejecución)
    docs = Path.home() / "Documents"
    docs.mkdir(exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix="e2e_domestico_", dir=docs))
    fotos = [sandbox / "vacaciones.jpg", sandbox / "cumple.jpg"]
    (sandbox / "factura.pdf").write_text("pdf")
    for f in fotos:
        f.write_text("jpg")
    dest = sandbox / "Imagenes"
    print(f"[1] Sandbox: {sandbox}  (2 jpg + 1 pdf)")

    # ── Ronda 1: el modelo decide la tool ────────────────────────────
    user_msg = (
        f"Mueve los archivos {fotos[0]} y {fotos[1]} a la carpeta {dest}. "
        "No toques el resto."
    )
    messages = [{"role": "user", "content": user_msg}]
    r1 = chat(messages, TOOLS)
    choice = r1["choices"][0]
    tool_calls = choice["message"]["tool_calls"]
    assert choice["finish_reason"] == "tool_calls", f"Sin tool_call: {choice}"
    call = tool_calls[0]
    fn = call["function"]
    args = json.loads(fn["arguments"])
    print(f"[2] Gemma eligió: {fn['name']}({json.dumps(args, ensure_ascii=False)[:120]}...)")
    assert fn["name"] == "file_organize", f"Tool equivocada: {fn['name']}"

    # ── Ejecución REAL de la herramienta ─────────────────────────────
    result = file_organize(files=args["files"], dest=args["dest"], dry_run=False)
    print(f"[3] Tool ejecutada: {result[:120]}")

    # ── Ronda 2: el modelo resume el resultado ───────────────────────
    messages += [
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": call["id"],
         "name": fn["name"], "content": result},
    ]
    r2 = chat(messages, TOOLS)
    final = r2["choices"][0]["message"]["content"]
    print(f"[4] Respuesta final: {(final or '')[:250]}")

    # ── Verificación en disco ────────────────────────────────────────
    moved = all((dest / f.name).exists() for f in fotos)
    untouched = (sandbox / "factura.pdf").exists()
    print(f"[5] Movidos: {moved} · PDF intacto: {untouched}")

    shutil.rmtree(sandbox, ignore_errors=True)
    if moved and untouched and final:
        print("\n✅ E2E DOMESTICO OK — decisión, ejecución real y resumen.")
        return 0
    print("\n❌ E2E FALLIDO")
    return 1


if __name__ == "__main__":
    sys.exit(main())
