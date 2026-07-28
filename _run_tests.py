"""
_run_tests.py
=============
Runner de tests organizado por entorno/contenedor.

Uso:
    python _run_tests.py            # corre los tests del entorno actual
    python _run_tests.py --dev      # tests sin GPU (Windows dev / cualquier máquina)
    python _run_tests.py --worker   # tests que necesitan torch/peft (worker container)
    python _run_tests.py --all      # todos (requiere torch+peft instalados)
    python _run_tests.py --list     # muestra qué test va en qué entorno

Arquitectura de contenedores:
    dev     → Python 3.13, Windows. Sin GPU, sin torch.
    serve   → Python 3.12, Linux. Sin GPU. FastAPI + llama-cpp.
    worker  → Python 3.11, Linux + CUDA. Torch + PEFT. Entrenamiento.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Tests que corren en CUALQUIER entorno (sin GPU, sin torch)
DEV_TESTS = [
    "tests/test_g2.py",           # DataDigestor — formatos exportación
    "tests/test_g4.py",           # DataDigestor — score_examples, validate
    "tests/test_g5.py",           # DataDigestor — from_docx, from_pdf
    "tests/test_s6.py",           # domestic_dataset_gen — generación plantillas
    "tests/test_domestic.py",     # domestic_tools + generate_dataset (sin GPU)
    "tests/test_s9.py",           # DataDigestor.from_user_profile + stack_adapters (mockeado)
    "tests/test_session_15mayo.py",  # to_multiturn, _fuzzy_path, from_api_spec, ContinualLearner
    "tests/test_server_oai.py",      # endpoint OpenAI: logging, tools, streaming, parser Gemma
    "tests/test_server_security.py", # auth /agent, CORS
    "tests/test_lineage_guard.py",   # guard de linaje en promoción GGUF
    "tests/test_log_quality.py",     # filtro de calidad del interaction_log
    "tests/test_reflection.py",      # pase de reflexión (feedback implícito) + fusión DPO
    "tests/test_digestor_distill.py", # Digestor: modos + destilación markdown (higiene)
    "tests/test_agent_robustness.py", # agente: guards de shell, entradas hostiles
    "tests/test_exporter.py",         # ExportManager (GGUF vía importorskip llama_cpp)
    "tests/test_stream_cancel.py",    # cancelación de streaming abandonado
]

# Tests que requieren torch/peft/GPU (worker container, Python 3.11 + CUDA)
WORKER_TESTS = [
    "tests/test_continual_e2e.py",   # ContinualLearner e2e con torch real
    "tests/test_dpo_pipeline.py",    # DPO pipeline con TRL
    "tests/test_e2e.py",             # pipeline completo (digest→train→export)
    "tests/test_behavioral.py",      # tests de comportamiento del agente (requiere modelo)
    "tests/test_digestor_completo.py",  # digestor con datasets reales
    "tests/test_vlm_pipeline.py",    # pipeline VLM (requiere torch+vision)
    "tests/test_gemma4_support.py",  # gemma4_unified: analyzer + trainer (usa torch.nn)
    "tests/test_vlm_mask_prompt.py", # collator VLM: completion-only loss (usa torch)
    "tests/test_vlm_keep_best.py",   # VLMTrainer: guarda la mejor época, no la última
]

PYTEST_FLAGS = [
    "-v",
    "--tb=short",
]


def _run(test_files: list[str], label: str) -> int:
    """Ejecuta pytest en los ficheros dados. Devuelve el exit code."""
    existing = [f for f in test_files if (ROOT / f).exists()]
    missing  = [f for f in test_files if not (ROOT / f).exists()]

    if missing:
        print(f"\n[{label}] Ficheros no encontrados (skipped):")
        for m in missing:
            print(f"    {m}")

    if not existing:
        print(f"\n[{label}] No hay tests disponibles.")
        return 0

    cmd = [sys.executable, "-m", "pytest"] + PYTEST_FLAGS + existing
    print(f"\n{'='*70}")
    print(f"[{label}] Ejecutando {len(existing)} fichero(s) de tests")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def _warn_unlisted() -> list[str]:
    """
    Avisa de ficheros test_*.py presentes en tests/ que no están en ningún
    manifiesto. Sin esto, un test nuevo queda huérfano y `--all` lo ignora en
    silencio (pasó con test_agent_robustness.py y test_exporter.py: 108 tests
    que el runner no corría, aunque el CI —pytest tests/— sí).
    """
    listed   = {Path(t).name for t in DEV_TESTS + WORKER_TESTS}
    on_disk  = {p.name for p in (ROOT / "tests").glob("test_*.py")}
    unlisted = sorted(on_disk - listed)

    if unlisted:
        print("\n" + "!" * 70)
        print("AVISO: ficheros de test NO listados en DEV_TESTS ni WORKER_TESTS.")
        print("Este runner NO los ejecuta. Añádelos al manifiesto que corresponda:")
        for u in unlisted:
            print(f"    tests/{u}")
        print("!" * 70)

    return unlisted


def _list_tests():
    print("\nTests por entorno:")
    print("\n  [DEV / cualquier máquina — sin GPU]")
    for t in DEV_TESTS:
        status = "✓" if (ROOT / t).exists() else "✗ (no encontrado)"
        print(f"    {status}  {t}")
    print("\n  [WORKER container — torch + CUDA requerido]")
    for t in WORKER_TESTS:
        status = "✓" if (ROOT / t).exists() else "✗ (no encontrado)"
        print(f"    {status}  {t}")


def main():
    args = sys.argv[1:]

    _warn_unlisted()

    if "--list" in args:
        _list_tests()
        return

    if "--worker" in args:
        rc = _run(WORKER_TESTS, "WORKER")
        sys.exit(rc)

    if "--all" in args:
        rc_dev    = _run(DEV_TESTS,    "DEV")
        rc_worker = _run(WORKER_TESTS, "WORKER")
        sys.exit(max(rc_dev, rc_worker))

    # Por defecto (y --dev): solo tests sin GPU
    rc = _run(DEV_TESTS, "DEV")
    sys.exit(rc)


if __name__ == "__main__":
    main()

