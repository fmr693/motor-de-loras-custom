#!/usr/bin/env python
"""
chequeo_activo.py — ¿cuánta señal de entrenamiento hemos acumulado?
===================================================================

El activo del proyecto es el dataset que va dejando el uso diario (ver
SESION.md, roadmap punto 2). Este script mide, EN SECO y sin GPU, cuánto
material de entrenamiento hay ya en el `interaction_log` y a qué distancia
estamos del umbral para el primer entrenamiento de comportamiento.

Umbrales decididos de antemano (SESION.md):
    ~300-500 pares DPO   →  entrenamiento de preferencia (DPO/ORPO)
    ~1-2k ejemplos SFT   →  entrenamiento supervisado

Uso:
    python scripts/chequeo_activo.py
    python scripts/chequeo_activo.py --log otra/ruta/interaction_log.jsonl

Standalone: solo lee ficheros (log + carpeta de reflexión). NO necesita el
serve ni la GPU. La reflexión FRESCA (que genera pares de corrección para DPO)
sí necesita el serve —`fabrica_loras reflect`—; este chequeo solo cuenta lo que
ya está en disco y avisa si la reflexión está vacía o vieja.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Umbrales del roadmap (rango bajo = "ya se puede probar", alto = "cómodo").
SFT_MIN, SFT_OBJ = 1000, 2000
DPO_MIN, DPO_OBJ = 300, 500


def barra(actual: int, objetivo: int, ancho: int = 30) -> str:
    llenos = min(ancho, int(ancho * actual / objetivo)) if objetivo else 0
    return "[" + "#" * llenos + "-" * (ancho - llenos) + "]"


def main() -> int:
    ap = argparse.ArgumentParser(description="Chequeo de señal de entrenamiento acumulada.")
    ap.add_argument("--log", default=str(ROOT / "logs" / "interaction_log.jsonl"),
                    help="Ruta al interaction_log.jsonl")
    ap.add_argument("--reflection-dir", default=str(ROOT / "datasets" / "reflection"),
                    help="Carpeta con la salida de 'reflect' (feedback implícito)")
    args = ap.parse_args()

    log = Path(args.log)
    if not log.exists():
        print(f"[chequeo] No existe el log: {log}")
        print("          Aún no se ha usado el stack, o la ruta es otra.")
        return 1

    import contextlib
    import io

    from motor import log_quality as lq
    from motor.dpo_trainer import DPOBuilder

    total = sum(1 for line in log.open(encoding="utf-8") if line.strip())

    # load_sft_examples y DPOBuilder.stats() imprimen su propia traza; las
    # silenciamos para que el informe salga limpio (el chequeo es la salida).
    refl = Path(args.reflection_dir)
    refl_arg = str(refl) if refl.exists() else None
    builder = DPOBuilder(log_path=str(log), reflection_dir=refl_arg)
    with contextlib.redirect_stdout(io.StringIO()):
        sft = lq.load_sft_examples(str(log))          # SFT limpio (filtro de 'learn')
        st = builder.stats()                          # DPO: feedback + reflexión
    pares = st["pairs_available"]

    # --- reflexión en disco (pares de corrección = la vía real a DPO) ---
    pairs_file = refl / "reflection_pairs.jsonl"
    n_refl_pairs = (
        sum(1 for line in pairs_file.open(encoding="utf-8") if line.strip())
        if pairs_file.exists() else 0
    )

    print("=" * 62)
    print("  CHEQUEO DE SEÑAL DE ENTRENAMIENTO  (en seco, sin GPU)")
    print("=" * 62)
    print(f"  Log: {log}")
    print(f"  Interacciones registradas : {total}")
    print(f"  Feedback explícito         : 👍 {st['positive']}   👎 {st['negative']}")
    print()
    print("  --- Vía SFT (supervisado) ---")
    print(f"  {barra(len(sft), SFT_OBJ)}  {len(sft)} / {SFT_MIN}-{SFT_OBJ}")
    print(f"  Ejemplos SFT limpios (tras dedup): {len(sft)}")
    print()
    print("  --- Vía DPO (preferencia) ---")
    print(f"  {barra(pares, DPO_OBJ)}  {pares} / {DPO_MIN}-{DPO_OBJ}")
    print(f"  Pares de preferencia disponibles: {pares}")
    print(f"  Pares de corrección de reflexión en disco: {n_refl_pairs}")

    print()
    print("  --- Veredicto ---")
    sft_listo = len(sft) >= SFT_MIN
    dpo_listo = pares >= DPO_MIN
    if sft_listo or dpo_listo:
        cual = []
        if sft_listo:
            cual.append(f"SFT ({len(sft)})")
        if dpo_listo:
            cual.append(f"DPO ({pares})")
        print(f"  ✅ Umbral alcanzado por: {', '.join(cual)}. Se puede entrenar.")
    else:
        falta_sft = SFT_MIN - len(sft)
        print(f"  ⏳ Aún no. Faltan ~{falta_sft} ejemplos SFT (o {DPO_MIN - pares} pares DPO).")
        print("     El caudal lo genera el USO diario vía Odysseus/Hermes.")

    # Avisos honestos sobre la vía DPO, que es la que suele quedarse a 0.
    if pares == 0 and (st["positive"] > 0 or st["negative"] > 0):
        print()
        print("  ⚠ Hay feedback 👍/👎 pero 0 pares DPO: se necesita el MISMO prompt")
        print("    con una respuesta buena y otra mala. Los pulgares sueltos sobre")
        print("    prompts distintos NO forman pares. La vía real son los pares de")
        print("    corrección de la reflexión (usuario reformula → error).")
    if n_refl_pairs == 0:
        print()
        print("  ⚠ Reflexión sin pares de corrección en disco. Para generarlos:")
        print("      fabrica_loras reflect --log logs/interaction_log.jsonl \\")
        print("        --out datasets/reflection    (NECESITA el serve arriba)")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
