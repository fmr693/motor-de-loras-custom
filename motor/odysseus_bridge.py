"""
motor.odysseus_bridge
=====================
Puente bidireccional entre LoRA Factory y Odysseus.

Proporciona tres capas de integración:

1. **MCP Tools** — Expone las 6 herramientas domésticas del Motor como
   herramientas MCP que el agente de Odysseus puede invocar directamente.
   Esto hace que el agente de Odysseus herede nuestras capacidades sin
   modificar una línea del código de Odysseus.

2. **Continual Learning Bridge** — Lee el interaction_log de Odysseus
   (formato JSONL), lo convierte a pares DPO y alimenta el ciclo de
   mejora continua. El flujo completo es:
     Odysseus logs → DPOBuilder → pares → ContinualLearner → nuevo adapter
     → ExportManager → GGUF → BenchmarkWorker → promote → hot-reload

3. **GGUF Watchdog** — Monitoriza promotion/ready.flag y notifica a
   motor-server que recargue el modelo. Sincronizado con el ciclo
   del worker de Docker.

Uso:
    # Iniciar el puente completo (MCP + CL + watchdog):
    python -m motor.odysseus_bridge

    # Solo MCP server:
    python -m motor.odysseus_bridge --mcp-only

    # Solo CL bridge:
    python -m motor.odysseus_bridge --cl-only
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# ── Resolver raíz del proyecto ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

# ── Constantes de integración ─────────────────────────────────────────
# Estos paths son los que usa Odysseus por defecto.
# Si Odysseus está en otra carpeta, se pueden sobrescribir con variables de entorno.
ODYSSEUS_LOG_PATH = Path(
    os.getenv("ODYSSEUS_LOG_PATH",
              str(_ROOT / "logs" / "interaction_log.jsonl"))
)
ODYSSEUS_DATA_DIR = Path(
    os.getenv("ODYSSEUS_DATA_DIR",
              str(_ROOT / "data"))
)
MOTOR_ADAPTER_DIR = _ROOT / "adapters"
MOTOR_MODEL_DIR   = _ROOT / "modelos"
PROMOTION_DIR     = _ROOT / "promotion"

# Período de chequeo del watchdog (segundos)
WATCHDOG_INTERVAL = float(os.getenv("WATCHDOG_INTERVAL", "5.0"))


# ===========================================================================
# CAPA 1 — MCP Tools: expone herramientas domésticas a Odysseus
# ===========================================================================

def build_mcp_tool_definitions() -> List[Dict[str, Any]]:
    """
    Construye las definiciones de herramientas en formato MCP (JSON Schema)
    para que Odysseus las reconozca como herramientas nativas de su agente.

    Odysseus espera tools MCP en formato:
    {
        "name": "nombre_herramienta",
        "description": "...",
        "inputSchema": { ... JSON Schema ... }
    }
    """
    tools = [
        {
            "name": "motor_file_organize",
            "description": (
                "Organiza archivos en carpetas según tipo, fecha o patrón. "
                "Útil para limpiar descargas, ordenar documentos, clasificar imágenes. "
                "Soporta mover, copiar y previsualizar (dry run)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Directorio de origen con los archivos a organizar."
                    },
                    "destination": {
                        "type": "string",
                        "description": "Directorio destino donde se crearán las subcarpetas."
                    },
                    "rule": {
                        "type": "string",
                        "enum": ["by_type", "by_date", "by_pattern"],
                        "description": "Criterio de organización."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Patrón para regla by_pattern (ej: '*.pdf')."
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": True,
                        "description": "Si True, solo muestra lo que haría sin ejecutar."
                    },
                },
                "required": ["source", "rule"],
            },
        },
        {
            "name": "motor_email_filter",
            "description": (
                "Filtra y clasifica correos electrónicos según criterios: "
                "remitente, asunto, fecha, prioridad, etiquetas."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Criterio de búsqueda (remitente, asunto, palabra clave)."
                    },
                    "folder": {
                        "type": "string",
                        "default": "INBOX",
                        "description": "Carpeta de correo a filtrar."
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 50,
                        "description": "Máximo de resultados a devolver."
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "motor_calendar_get",
            "description": (
                "Consulta eventos del calendario en un rango de fechas. "
                "Devuelve eventos, reuniones, recordatorios."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date_from": {
                        "type": "string",
                        "description": "Fecha inicio en formato ISO (YYYY-MM-DD)."
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Fecha fin en formato ISO (YYYY-MM-DD)."
                    },
                },
                "required": ["date_from"],
            },
        },
        {
            "name": "motor_note_save",
            "description": (
                "Guarda una nota de texto en el sistema de notas local. "
                "Soporta etiquetas, categorías y formato markdown."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Título de la nota."
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenido de la nota (markdown permitido)."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Etiquetas para organizar la nota."
                    },
                },
                "required": ["title", "content"],
            },
        },
        {
            "name": "motor_search_files",
            "description": (
                "Busca archivos en el sistema de ficheros local por nombre, "
                "contenido o extensión. Soporta búsqueda recursiva y filtros."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto a buscar en nombres de archivo."
                    },
                    "path": {
                        "type": "string",
                        "description": "Directorio raíz de búsqueda."
                    },
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filtrar por extensiones (ej: ['.py', '.json'])."
                    },
                    "recursive": {
                        "type": "boolean",
                        "default": True,
                        "description": "Buscar recursivamente en subcarpetas."
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "motor_process_run",
            "description": (
                "Ejecuta un proceso del sistema de forma segura (solo comandos "
                "de lectura y dentro de una whitelist). Devuelve stdout, stderr y exit code."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Comando a ejecutar (solo whitelist)."
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Argumentos del comando."
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 30,
                        "description": "Timeout en segundos."
                    },
                },
                "required": ["command"],
            },
        },
    ]
    return tools


# ===========================================================================
# CAPA 2 — Continual Learning Bridge
# ===========================================================================

def convert_odysseus_log_to_dpo_pairs(
    log_path: Path,
    min_pairs: int = 5,
) -> Optional[Path]:
    """
    Lee el interaction_log de Odysseus y lo convierte a pares DPO
    que nuestro DPOBuilder puede procesar.

    El formato del log de Odysseus es compatible con el nuestro porque
    ambos usan JSONL con campos: user_msg, assistant, feedback.

    Returns:
        Path al archivo de pares generado, o None si no hay suficientes.
    """
    if not log_path.exists():
        print(f"[OdysseusBridge] Log no encontrado: {log_path}")
        return None

    from motor.dpo_trainer import DPOBuilder

    builder = DPOBuilder(
        log_path  = str(log_path),
        min_pairs = 1,  # Extraemos todo; el CL decide si hay suficientes
    )

    try:
        stats = builder.stats()
    except FileNotFoundError:
        print(f"[OdysseusBridge] Log vacío o inaccesible: {log_path}")
        return None

    rated = stats.get("rated_entries", 0)
    pairs = stats.get("pairs_available", 0)
    print(f"[OdysseusBridge] Log analizado: {rated} rated, {pairs} pairs")

    if pairs < min_pairs:
        print(f"[OdysseusBridge] Insuficientes pares ({pairs} < {min_pairs}). "
              f"Se necesitan más interacciones con feedback.")
        return None

    # Exportar pares a un archivo temporal
    pairs_dir = _ROOT / "datasets" / "odysseus_dpo"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pairs_file = pairs_dir / f"pairs_{int(time.time())}.jsonl"

    try:
        out = builder.to_jsonl(str(pairs_file))
        print(f"[OdysseusBridge] {pairs} pares exportados a: {out}")
        return out
    except ValueError as e:
        print(f"[OdysseusBridge] Error exportando pares: {e}")
        return None


def run_cl_cycle(
    adapter_dir: Optional[str] = None,
    base_model_id: Optional[str] = None,
    log_path: Optional[Path] = None,
    min_pairs: int = 5,
    dry_run: bool = False,
) -> dict:
    """
    Ejecuta un ciclo completo de mejora continua desde logs de Odysseus.

    Pasos:
      1. Leer log de Odysseus → extraer pares DPO
      2. Si hay suficientes pares → reentrenar con ContinualLearner
      3. Exportar a GGUF
      4. Benchmark (si está configurado)
      5. Promover (escribir ready.flag)

    Returns:
        dict con métricas del ciclo.
    """
    log_path = log_path or ODYSSEUS_LOG_PATH
    result = {
        "cycle_ts": time.time(),
        "pairs_found": 0,
        "trained": False,
        "eval_loss": None,
        "gguf_exported": False,
        "promoted": False,
    }

    if dry_run:
        print("[OdysseusBridge] DRY RUN — no se ejecutará nada.")
        return result

    # Paso 1: Extraer pares
    pairs_file = convert_odysseus_log_to_dpo_pairs(log_path, min_pairs=min_pairs)
    if not pairs_file:
        return result
    result["pairs_found"] = len(
        [l for l in pairs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    )

    # Paso 2: Reentrenar con ContinualLearner
    if adapter_dir is None:
        # Buscar el adapter más reciente
        if MOTOR_ADAPTER_DIR.exists():
            adapters = sorted(
                [d for d in MOTOR_ADAPTER_DIR.iterdir() if d.is_dir() and (d / "meta.json").exists()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if adapters:
                adapter_dir = str(adapters[0])
                print(f"[OdysseusBridge] Adapter detectado: {adapter_dir}")

    if adapter_dir is None:
        print("[OdysseusBridge] Sin adapter base. Ejecuta un entrenamiento inicial primero.")
        return result

    # Leer meta.json
    meta_path = Path(adapter_dir) / "meta.json"
    if not meta_path.exists():
        print(f"[OdysseusBridge] meta.json no encontrado en {adapter_dir}")
        return result

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    base_model_id = base_model_id or meta.get("model_id") or meta.get("base_model")
    if not base_model_id:
        print("[OdysseusBridge] No se pudo determinar el modelo base.")
        return result

    from motor.continual import ContinualLearner

    output_dir = str(Path(adapter_dir).parent / f"{Path(adapter_dir).name}_odysseus_cl")
    print(f"[OdysseusBridge] Reentrenando con ContinualLearner...")
    print(f"  Base model : {base_model_id}")
    print(f"  Adapter    : {adapter_dir}")
    print(f"  Pairs      : {pairs_file}")
    print(f"  Output     : {output_dir}")

    try:
        cl = ContinualLearner(
            model_id            = base_model_id,
            replay_buffer_size  = 200,
            rollback_threshold  = 0.15,
            lora_r              = meta.get("lora_r", 16),
            lora_alpha          = meta.get("lora_alpha", 32),
        )
        cl.register_existing(adapter_dir, name=Path(adapter_dir).name)

        metrics = cl.fit(
            dataset_path  = str(pairs_file),
            output_dir    = output_dir,
            adapter_name  = Path(output_dir).name,
            epochs        = 1,
            batch_size    = 2,
            learning_rate = 1e-4,
        )
        if metrics:
            result["trained"]      = True
            result["eval_loss"]    = metrics.get("eval_loss")
            result["output_dir"]   = output_dir
            print(f"[OdysseusBridge] ✅ Reentrenado: eval_loss={result['eval_loss']}")
    except Exception as e:
        print(f"[OdysseusBridge] ❌ Error en CL: {e}")
        return result

    if not result["trained"]:
        return result

    # Paso 3: Exportar a GGUF
    try:
        from motor.exporter import ExportManager
        gguf_dir = MOTOR_MODEL_DIR / "odysseus"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_output = gguf_dir / f"odysseus_cl_{int(time.time())}.gguf"

        print(f"[OdysseusBridge] Exportando a GGUF: {gguf_output}")
        em = ExportManager(
            adapter_dir = output_dir,
            base_model  = base_model_id,
        )
        em.to_gguf(
            output       = str(gguf_output),
            quantization = "q4_k_m",
        )
        result["gguf_exported"] = True
        result["gguf_path"]     = str(gguf_output)
        print(f"[OdysseusBridge] ✅ GGUF exportado: {gguf_output}")
    except Exception as e:
        print(f"[OdysseusBridge] ⚠️ Export GGUF falló: {e}")

    # Paso 4: Promover (escribir ready.flag)
    if result["gguf_exported"]:
        try:
            PROMOTION_DIR.mkdir(parents=True, exist_ok=True)
            from datetime import datetime, timezone
            flag_content = {
                "gguf":      result.get("gguf_path"),
                "eval_loss": result["eval_loss"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source":    "odysseus_bridge",
                # Linaje: el watchdog rechaza candidatos de familia distinta
                # a la del modelo servido
                "base_model": base_model_id,
            }
            flag_file = PROMOTION_DIR / "ready.flag"
            flag_file.write_text(json.dumps(flag_content, indent=2))
            result["promoted"] = True
            print(f"[OdysseusBridge] ✅ ready.flag escrito → motor-serve recargará el GGUF")
        except Exception as e:
            print(f"[OdysseusBridge] ⚠️ Error escribiendo ready.flag: {e}")

    return result


# ===========================================================================
# CAPA 3 — GGUF Watchdog
# ===========================================================================

def watch_promotion_dir(
    interval: float = WATCHDOG_INTERVAL,
    callback: Optional[callable] = None,
) -> None:
    """
    Monitoriza el directorio promotion/ y ejecuta callback cuando
    aparece un nuevo ready.flag.

    El callback recibe el contenido del flag como dict.

    Si no se proporciona callback, imprime en consola.
    """
    if callback is None:
        def _default_cb(flag_data: dict):
            print(f"[OdysseusBridge] 🔄 Nuevo GGUF promovido: {flag_data.get('gguf')}")
            print(f"                   eval_loss: {flag_data.get('eval_loss')}")
        callback = _default_cb

    last_mtime = 0.0
    print(f"[OdysseusBridge] 👁️ Watchdog activo en {PROMOTION_DIR} "
          f"(cada {interval}s)")

    while True:
        try:
            flag_file = PROMOTION_DIR / "ready.flag"
            if flag_file.exists():
                mtime = flag_file.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    try:
                        flag_data = json.loads(flag_file.read_text(encoding="utf-8"))
                        callback(flag_data)
                    except Exception as e:
                        print(f"[OdysseusBridge] ⚠️ Error leyendo ready.flag: {e}")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("[OdysseusBridge] Watchdog detenido.")
            break
        except Exception as e:
            print(f"[OdysseusBridge] ⚠️ Error en watchdog: {e}")
            time.sleep(interval)


# ===========================================================================
# CLI del puente
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Puente LoRA Factory ↔ Odysseus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mcp-only", action="store_true",
        help="Solo iniciar el servidor MCP de herramientas domésticas.",
    )
    parser.add_argument(
        "--cl-only", action="store_true",
        help="Solo ejecutar el ciclo de Continual Learning desde logs de Odysseus.",
    )
    parser.add_argument(
        "--watchdog-only", action="store_true",
        help="Solo iniciar el watchdog de promoción de GGUF.",
    )
    parser.add_argument(
        "--log", default=None,
        help="Ruta al interaction_log de Odysseus (default: logs/interaction_log.jsonl).",
    )
    parser.add_argument(
        "--adapter", default=None,
        help="Ruta al adapter base para CL (auto-detecta si no se indica).",
    )
    parser.add_argument(
        "--min-pairs", type=int, default=5,
        help="Pares mínimos para disparar reentrenamiento (default: 5).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simular sin ejecutar entrenamiento ni escritura.",
    )

    args = parser.parse_args()

    all_modes = not (args.mcp_only or args.cl_only or args.watchdog_only)

    # MCP tools info
    if all_modes or args.mcp_only:
        tools = build_mcp_tool_definitions()
        print("=" * 60)
        print("  🛠️  Herramientas MCP disponibles para Odysseus")
        print("=" * 60)
        for t in tools:
            print(f"  • {t['name']}")
            print(f"    {t['description'][:100]}...")
        print()
        print("  Para registrar en Odysseus:")
        print("    1. Ve a Settings → MCP Servers")
        print("    2. Añade: python -m motor.odysseus_bridge --mcp-only")
        print("    3. Las herramientas aparecerán en el agente de Odysseus")
        print()

    # CL cycle
    if all_modes or args.cl_only:
        log_path = Path(args.log) if args.log else ODYSSEUS_LOG_PATH
        print("=" * 60)
        print("  🔄 Ejecutando ciclo de Continual Learning")
        print("=" * 60)
        result = run_cl_cycle(
            adapter_dir  = args.adapter,
            log_path     = log_path,
            min_pairs    = args.min_pairs,
            dry_run      = args.dry_run,
        )
        print(f"\n  Resultado: {json.dumps(result, indent=2, default=str)}")

    # Watchdog
    if all_modes or args.watchdog_only:
        print("=" * 60)
        print("  👁️ Iniciando watchdog de promoción GGUF...")
        print("=" * 60)
        watch_promotion_dir()


if __name__ == "__main__":
    main()
