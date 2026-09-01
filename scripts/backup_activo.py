"""
backup_activo.py
================
Copia de seguridad del ACTIVO del proyecto: el dato, que es lo irreemplazable.

Contexto (ver PRUEBAS_ESTRES.md): la tesis del proyecto es que el activo duradero
es el DATASET, no el adapter. Pero el log canónico y los datasets viven en carpetas
runtime, gitignoradas y sin copia. La ronda 4 de estrés demostró que el log puede
perder datos bajo concurrencia (lo blindamos con _LOG_LOCK), pero eso no protege
contra un disco muerto. Este script sí.

Qué copia (pequeño e irreemplazable):
  - logs/*.jsonl        el log canónico + rotados (interaction_log)
  - datasets/           datasets SFT + salidas de reflexión (DPO)
  - SESION.md           la bitácora de continuidad (gitignorada)
  - MEMORY.md + memoria persistente, si se le indica --memory-dir

NO copia por defecto (grande y REPRODUCIBLE desde el dato):
  - adapters/           612 MB de adapters entrenados -> --adapters para incluirlos
  - modelos/            GGUF + mmproj -> re-descargables, nunca se copian

Uso:
    python scripts/backup_activo.py                 # backup a la carpeta por defecto
    python scripts/backup_activo.py --dest D:/backups/motor   # destino explícito
    python scripts/backup_activo.py --adapters      # incluye adapters/ (pesado)
    python scripts/backup_activo.py --keep 12       # conserva los últimos 12 backups

Destino por defecto: MOTOR_BACKUP_DIR o ../_backups_motor  (junto al repo).
AVISO: por defecto copia al MISMO disco. Para proteccion real, apunta --dest a
otro disco o una carpeta sincronizada a la nube (OneDrive/Drive).

Determinista, stdlib pura, nunca borra el original. Pensado para tarea programada
semanal (ver scripts/README_scripts.md). Salida ASCII a proposito: los caracteres
unicode revientan en la consola cp1252 de Windows (leccion de las pruebas de estres).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
from pathlib import Path

# Consola de Windows en cp1252 rompe con acentos/unicode; forzar UTF-8 si se puede.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _copy_into(src: Path, dst_dir: Path) -> int:
    """Copia src (archivo o carpeta) dentro de dst_dir. Devuelve bytes copiados."""
    if not src.exists():
        print(f"  - {src.name}: no existe, se omite")
        return 0
    target = dst_dir / src.name
    if src.is_dir():
        shutil.copytree(src, target, dirs_exist_ok=True)
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        size = target.stat().st_size
    print(f"  [ok] {src.name}  ({_human(size)})")
    return size


def _prune(dest_root: Path, keep: int) -> None:
    """Conserva solo los `keep` backups más recientes (por nombre = timestamp)."""
    backups = sorted(
        (p for p in dest_root.glob("backup_*") if p.is_dir()),
        reverse=True,
    )
    for old in backups[keep:]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"  - purgado backup antiguo: {old.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Backup del activo (dato) de LoRA Factory.")
    default_dest = os.environ.get("MOTOR_BACKUP_DIR", str(ROOT.parent / "_backups_motor"))
    ap.add_argument("--dest", default=default_dest,
                    help=f"Carpeta destino de los backups (def: {default_dest}).")
    ap.add_argument("--adapters", action="store_true",
                    help="Incluir adapters/ (pesado; reproducible desde el dato).")
    ap.add_argument("--memory-dir", default=None,
                    help="Carpeta de memoria persistente a incluir (MEMORY.md + *.md).")
    ap.add_argument("--keep", type=int, default=8,
                    help="Numero de backups a conservar (def: 8).")
    args = ap.parse_args()

    dest_root = Path(args.dest)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = dest_root / f"backup_{stamp}"
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[backup] LoRA Factory - activo -> {dst}")
    total = 0

    # Log canónico + rotados
    logs = ROOT / "logs"
    if logs.is_dir():
        jsonl = list(logs.glob("*.jsonl"))
        if jsonl:
            (dst / "logs").mkdir(exist_ok=True)
            for f in jsonl:
                total += _copy_into(f, dst / "logs")
        else:
            print("  - logs/: sin .jsonl, se omite")

    # Datasets (SFT + reflexión)
    total += _copy_into(ROOT / "datasets", dst)

    # Bitácora de continuidad (gitignorada)
    total += _copy_into(ROOT / "SESION.md", dst)

    # Memoria persistente (opcional)
    if args.memory_dir:
        total += _copy_into(Path(args.memory_dir), dst / "memory")

    # Adapters (opcional, pesado)
    if args.adapters:
        total += _copy_into(ROOT / "adapters", dst)

    # Manifiesto del backup
    (dst / "MANIFEST.txt").write_text(
        f"Backup de LoRA Factory\n"
        f"Fecha: {dt.datetime.now().isoformat()}\n"
        f"Origen: {ROOT}\n"
        f"Tamano total: {_human(total)}\n"
        f"Incluye adapters: {args.adapters}\n",
        encoding="utf-8",
    )

    _prune(dest_root, args.keep)

    print(f"[backup] OK - {_human(total)} copiados a {dst}")
    if str(dest_root).lower().startswith(str(ROOT.anchor).lower()) and \
       Path(dest_root).drive.lower() == ROOT.drive.lower():
        print("[backup] AVISO: el destino esta en el MISMO disco que el proyecto. "
              "Para proteccion real, usa --dest en otro disco o una carpeta en la nube.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
