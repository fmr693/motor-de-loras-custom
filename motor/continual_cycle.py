"""
motor.continual_cycle
=====================
Ciclo de mejora continua del agente doméstico.

Orquesta los 5 pasos que convierten logs de usuario en un modelo mejorado:

  1. DataDigestor      — logs/interaction_log.jsonl → dataset JSONL limpio
  2. ContinualLearner  — reentrenamiento incremental con replay buffer
  3. ExportManager     — merge + cuantización Q4_K_M → modelos/candidate.gguf
  4. BenchmarkWorker   — valida el candidato con tareas estándar
  5. Promoción         — escribe promotion/ready.flag para el watchdog del servidor

Uso
---
  # Desde docker-compose (CMD del Dockerfile.train):
  python -m motor.continual_cycle

  # Manual con argumentos:
  python -m motor.continual_cycle --min-examples 50 --dry-run

  # Solo un paso:
  python -m motor.continual_cycle --only-step digest
  python -m motor.continual_cycle --only-step train
  python -m motor.continual_cycle --only-step export
  python -m motor.continual_cycle --only-step benchmark
  python -m motor.continual_cycle --only-step promote

Variables de entorno
--------------------
  BASE_MODEL_ID   — modelo HuggingFace base (default: Qwen/Qwen2.5-7B-Instruct)
  ADAPTER_NAME    — nombre del adapter en adapters/ (default: domestic_current)
  MIN_EXAMPLES    — mínimo de ejemplos en el dataset para entrenar (default: 50)
  DRY_RUN         — si "1", simula sin escribir nada (default: 0)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Rutas (relativas al directorio de trabajo /app en Docker)
# ─────────────────────────────────────────────────────────────────────────────
ROOT          = Path(os.getenv("APP_ROOT", "."))
LOGS_DIR      = ROOT / "logs"
DATASETS_DIR  = ROOT / "datasets"
ADAPTERS_DIR  = ROOT / "adapters"
MODELOS_DIR   = ROOT / "modelos"
PROMOTION_DIR = ROOT / "promotion"

LOG_FILE       = LOGS_DIR / "interaction_log.jsonl"
CANDIDATE_GGUF = MODELOS_DIR / "candidate.gguf"
CURRENT_GGUF   = MODELOS_DIR / "current.gguf"
READY_FLAG     = PROMOTION_DIR / "ready.flag"

# ─────────────────────────────────────────────────────────────────────────────
# Configuración por defecto (sobreescribible con env vars)
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_NAME  = os.getenv("ADAPTER_NAME",  "domestic_current")
MIN_EXAMPLES  = int(os.getenv("MIN_EXAMPLES", "50"))
DRY_RUN       = os.getenv("DRY_RUN", "0") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _abort(reason: str) -> None:
    print(f"\n[CICLO] ABORTADO — {reason}")
    sys.exit(1)


def _next_dataset_path() -> Path:
    """Devuelve datasets/domestic_vN.jsonl con N = siguiente disponible."""
    existing = sorted(DATASETS_DIR.glob("domestic_v*.jsonl"))
    if not existing:
        return DATASETS_DIR / "domestic_v1.jsonl"
    last = existing[-1]
    n = int(last.stem.split("_v")[-1]) + 1
    return DATASETS_DIR / f"domestic_v{n}.jsonl"


def _next_adapter_path() -> Path:
    """Devuelve adapters/domestic_vN/ con N = siguiente disponible."""
    existing = sorted([d for d in ADAPTERS_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("domestic_v")])
    if not existing:
        return ADAPTERS_DIR / "domestic_v1"
    last = existing[-1]
    n = int(last.name.split("_v")[-1]) + 1
    return ADAPTERS_DIR / f"domestic_v{n}"


def _latest_adapter_path() -> Optional[Path]:
    """Devuelve el adapter más reciente en adapters/domestic_vN/."""
    existing = sorted([d for d in ADAPTERS_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("domestic_v")])
    return existing[-1] if existing else None


# ─────────────────────────────────────────────────────────────────────────────
# Paso 1 — DataDigestor
# ─────────────────────────────────────────────────────────────────────────────

def step_digest(dry_run: bool = False) -> Path:
    """
    Lee logs/interaction_log.jsonl y produce el siguiente dataset versionado.
    Semáforo: ROJO → abortar, AMARILLO → warning pero continuar, VERDE → ok.
    """
    _header("PASO 1 — DataDigestor: logs → dataset")

    if not LOG_FILE.exists():
        _abort(f"No existe el log de interacciones: {LOG_FILE}")

    # Contar líneas del log para diagnóstico rápido
    with open(LOG_FILE, encoding="utf-8") as f:
        n_logs = sum(1 for _ in f)
    print(f"[Digestor] Logs disponibles: {n_logs} interacciones")

    if n_logs < MIN_EXAMPLES:
        _abort(
            f"Solo {n_logs} logs disponibles — mínimo {MIN_EXAMPLES}. "
            "Acumula más interacciones antes de reentrenar."
        )

    from motor.digestor import DataDigestor

    dataset_path = _next_dataset_path()
    print(f"[Digestor] Dataset destino: {dataset_path}")

    if dry_run:
        print("[Digestor] DRY RUN — no se escribe nada.")
        return dataset_path

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    d = DataDigestor()
    d.from_conversations(str(LOG_FILE))

    # Semáforo de calidad
    report = d.validate()
    level  = report.get("level", "VERDE")
    print(f"[Digestor] Semáforo: {level}")
    for warn in report.get("warnings", []):
        print(f"  ⚠  {warn}")

    if level == "ROJO":
        _abort(f"Semáforo ROJO — dataset no apto para entrenamiento. {report.get('warnings')}")

    d.to_jsonl(str(dataset_path))
    n_examples = sum(1 for _ in open(dataset_path, encoding="utf-8"))
    print(f"[Digestor] Dataset generado: {n_examples} ejemplos → {dataset_path}")
    return dataset_path


# ─────────────────────────────────────────────────────────────────────────────
# Paso 2 — ContinualLearner
# ─────────────────────────────────────────────────────────────────────────────

def step_train(dataset_path: Path, dry_run: bool = False) -> Path:
    """
    Reentrenamiento incremental partiendo del adapter más reciente.
    Incluye replay buffer y rollback automático si hay regresión >15%.
    """
    _header("PASO 2 — ContinualLearner: reentrenamiento incremental")

    base_adapter = _latest_adapter_path()
    new_adapter  = _next_adapter_path()

    print(f"[Train] Modelo base:    {BASE_MODEL_ID}")
    print(f"[Train] Adapter previo: {base_adapter or 'ninguno (primer entrenamiento)'}")
    print(f"[Train] Adapter nuevo:  {new_adapter}")

    if dry_run:
        print("[Train] DRY RUN — no se entrena.")
        return new_adapter

    from motor.continual import ContinualLearner

    cl = ContinualLearner(
        base_model_id      = BASE_MODEL_ID,
        adapter_output_dir = str(new_adapter),
        replay_buffer_size = 200,
        rollback_threshold = 0.15,
    )

    result = cl.fit(
        new_dataset_path    = str(dataset_path),
        previous_adapter    = str(base_adapter) if base_adapter else None,
    )

    if result.get("rolled_back"):
        print(
            f"[Train] ROLLBACK aplicado — regresión {result['regression_pct']:.1%} > 15%. "
            f"Se conserva el adapter anterior."
        )
        # Rollback: el adapter nuevo se descarta, devolvemos el anterior
        shutil.rmtree(str(new_adapter), ignore_errors=True)
        return base_adapter  # type: ignore[return-value]

    print(f"[Train] Entrenamiento completado. eval_loss={result.get('eval_loss', '?')}")
    return new_adapter


# ─────────────────────────────────────────────────────────────────────────────
# Paso 3 — ExportManager
# ─────────────────────────────────────────────────────────────────────────────

def step_export(adapter_path: Path, dry_run: bool = False) -> Path:
    """
    Fusiona base + adapter, cuantiza a Q4_K_M y produce candidate.gguf.
    """
    _header("PASO 3 — ExportManager: merge + Q4_K_M → candidate.gguf")

    print(f"[Export] Adapter: {adapter_path}")
    print(f"[Export] Destino: {CANDIDATE_GGUF}")

    if dry_run:
        print("[Export] DRY RUN — no se exporta.")
        return CANDIDATE_GGUF

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    from motor.exporter import ExportManager

    em = ExportManager(
        adapter_path  = str(adapter_path),
        base_model_id = BASE_MODEL_ID,
    )
    em.export_gguf(
        output_path  = str(CANDIDATE_GGUF),
        quantization = "q4_k_m",
    )

    size_gb = CANDIDATE_GGUF.stat().st_size / 1e9
    print(f"[Export] GGUF generado: {size_gb:.2f} GB → {CANDIDATE_GGUF}")
    return CANDIDATE_GGUF


# ─────────────────────────────────────────────────────────────────────────────
# Paso 4 — Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def step_benchmark(candidate_path: Path, dry_run: bool = False) -> bool:
    """
    Valida el GGUF candidato con el conjunto de tareas estándar.
    Devuelve True si supera el umbral mínimo.
    """
    _header("PASO 4 — Benchmark: validación del candidato")

    if dry_run:
        print("[Benchmark] DRY RUN — se asume que el candidato es válido.")
        return True

    from motor.benchmark_worker import run_benchmark

    passed, report = run_benchmark(str(candidate_path))
    print(f"[Benchmark] Resultado: {'✅ PASADO' if passed else '❌ FALLADO'}")
    for task_name, ok in report.items():
        print(f"  {'✅' if ok else '❌'} {task_name}")

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Paso 5 — Promoción
# ─────────────────────────────────────────────────────────────────────────────

def step_promote(dry_run: bool = False) -> None:
    """
    Escribe promotion/ready.flag para que el watchdog del servidor recargue.
    """
    _header("PASO 5 — Promoción: señal al servidor")

    if dry_run:
        print("[Promover] DRY RUN — no se escribe el flag.")
        return

    PROMOTION_DIR.mkdir(parents=True, exist_ok=True)
    READY_FLAG.write_text(
        json.dumps({
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # Linaje del candidato: el watchdog del servidor rechaza la
            # promoción si la familia no coincide con el modelo servido
            "base_model":  BASE_MODEL_ID,
        }),
        encoding="utf-8",
    )
    print(f"[Promover] Flag escrito: {READY_FLAG}")
    print("[Promover] El servidor recargará el modelo en ≤30 segundos.")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_cycle(only_step: Optional[str] = None, dry_run: bool = False) -> None:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  CICLO DE MEJORA CONTINUA")
    print(f"  Base model : {BASE_MODEL_ID}")
    print(f"  Min ejemplos: {MIN_EXAMPLES}")
    print(f"  Dry run    : {dry_run}")
    print(f"{'='*60}")

    # ── Informe de hardware ──────────────────────────────────────────────────
    try:
        from motor.hardware import detect_hardware
        hw = detect_hardware()
        print(hw)
        print(f"  Perfil entrenamiento : {hw.training_profile}")
        print(f"  Perfil inferencia    : {hw.inference_profile}")
        print(f"{'='*60}")
    except Exception as _hw_err:
        print(f"  [Hardware] No se pudo detectar: {_hw_err}")
    # ────────────────────────────────────────────────────────────────────────

    steps = {"digest", "train", "export", "benchmark", "promote"}
    if only_step and only_step not in steps:
        _abort(f"--only-step debe ser uno de: {sorted(steps)}")

    # Paso 1
    if not only_step or only_step == "digest":
        dataset_path = step_digest(dry_run)
    else:
        dataset_path = _next_dataset_path()   # para pasos subsiguientes

    if only_step == "digest":
        return

    # Paso 2
    if not only_step or only_step == "train":
        adapter_path = step_train(dataset_path, dry_run)
    else:
        adapter_path = _latest_adapter_path() or _abort("No hay adapter disponible")  # type: ignore[arg-type]

    if only_step == "train":
        return

    # Paso 3
    if not only_step or only_step == "export":
        candidate_path = step_export(adapter_path, dry_run)  # type: ignore[arg-type]
    else:
        candidate_path = CANDIDATE_GGUF

    if only_step == "export":
        return

    # Paso 4
    if not only_step or only_step == "benchmark":
        passed = step_benchmark(candidate_path, dry_run)
        if not passed:
            _abort(
                "El candidato no superó el benchmark. "
                "Se conserva el modelo actual sin cambios."
            )

    if only_step == "benchmark":
        return

    # Paso 5
    if not only_step or only_step == "promote":
        step_promote(dry_run)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  CICLO COMPLETADO en {elapsed/60:.1f} min")
    print(f"{'='*60}\n")


def main() -> None:
    global MIN_EXAMPLES
    parser = argparse.ArgumentParser(
        description="Ciclo de mejora continua del agente doméstico"
    )
    parser.add_argument(
        "--only-step",
        choices=["digest", "train", "export", "benchmark", "promote"],
        help="Ejecutar solo un paso del ciclo",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=MIN_EXAMPLES,
        help=f"Mínimo de ejemplos en el log para entrenar (default: {MIN_EXAMPLES})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular sin escribir nada",
    )
    args = parser.parse_args()

    MIN_EXAMPLES = args.min_examples

    run_cycle(only_step=args.only_step, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
