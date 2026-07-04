"""
test_vlm_pipeline.py
====================
Prueba del pipeline VLM: from_images_folder_vlm() + VLMTrainer.

Genera imágenes sintéticas de colores y las organiza en carpetas
etiquetadas. Luego genera el dataset .jsonl listo para VLMTrainer.

Ejecutar:
  python test_vlm_pipeline.py          # genera el dataset
  python test_vlm_pipeline.py --train  # entrena (requiere GPU + CUDA)
"""

import sys, os, tempfile
from pathlib import Path

from PIL import Image

RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
BLUE = "\033[94m"; BOLD = "\033[1m"; RESET = "\033[0m"

TMP = Path(tempfile.mkdtemp(prefix="vlm_test_"))
OUT = Path(__file__).resolve().parent / "test_vlm_output"
OUT.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{BLUE}╔════════════════════════════════════════════════╗{RESET}")
print(f"{BOLD}{BLUE}║  TEST VLM — Dataset + Entrenamiento           ║{RESET}")
print(f"{BOLD}{BLUE}╚════════════════════════════════════════════════╝{RESET}")

# Paso 1: Crear imágenes sintéticas por categoría
print(f"\n{GREEN}▶ Paso 1: Generando imágenes sintéticas...{RESET}")

categories = {
    "rojo":     (255, 0, 0),
    "verde":    (0, 255, 0),
    "azul":     (0, 0, 255),
    "amarillo": (255, 255, 0),
}
images_folder = TMP / "imagenes"

for cat, color in categories.items():
    cat_dir = images_folder / cat
    cat_dir.mkdir(parents=True, exist_ok=True)
    for i in range(15):  # 15 imágenes por categoría
        img = Image.new("RGB", (100, 100), color)
        # Añadir variación: cambiar ligeramente el tono
        r, g, b = color
        r = min(255, max(0, r + (i * 3) % 30))
        g = min(255, max(0, g + (i * 5) % 30))
        b = min(255, max(0, b + (i * 7) % 30))
        img = Image.new("RGB", (100, 100), (r, g, b))
        img.save(cat_dir / f"{cat}_{i:02d}.png")

print(f"  {GREEN}✓{RESET} {sum(1 for _ in images_folder.rglob('*.png'))} imágenes en {len(categories)} categorías")
print(f"    {images_folder}")

# Paso 2: Generar dataset VLM
print(f"\n{GREEN}▶ Paso 2: Generando dataset VLM...{RESET}")

from motor.digestor import DataDigestor

d = DataDigestor(
    task="¿De qué color es esta imagen? Responde SOLO con el nombre del color: rojo, verde, azul o amarillo.",
    auto_enrich=True,
)
d.from_images_folder_vlm(
    folder=images_folder,
    question="¿De qué color es esta imagen?",
    label_from_subfolder=True,
)

print(f"  {GREEN}✓{RESET} {len(d._examples)} ejemplos generados")

# Validar
result = d.validate(verbose=True)

# Exportar
jsonl_path = OUT / "colores_vlm.jsonl"
n = d.to_jsonl(str(jsonl_path), shuffle=True)
print(f"\n  {GREEN}✓{RESET} Dataset exportado: {jsonl_path} ({n} ejemplos)")

# Mostrar primer ejemplo
import json
with open(jsonl_path, encoding="utf-8") as f:
    first = json.loads(f.readline())
print(f"\n  {BOLD}Primer ejemplo:{RESET}")
msgs = first["messages"]
for m in msgs:
    role = m["role"].upper()
    content = m.get("content", "")
    if isinstance(content, list):
        parts = []
        for p in content:
            t = p.get("type", "?")
            if t == "image":
                parts.append(f"[IMAGEN: {Path(p.get('image','')).name}]")
            else:
                parts.append(str(p.get("text", ""))[:80])
        content = " | ".join(parts)
    else:
        content = str(content)[:100]
    print(f"    [{role}] {content}")

# Paso 3: Instrucciones para entrenar
print(f"\n{BOLD}{YELLOW}{'═'*50}{RESET}")
print(f"{BOLD}{YELLOW}  Para entrenar en el servidor:{RESET}")
print(f"{BOLD}{YELLOW}{'═'*50}{RESET}")
print(f"""
  # 1. Subir el dataset al servidor:
  scp {jsonl_path} felipe@192.168.1.45:~/Proyecto_V3/datasets/colores_vlm.jsonl

  # 2. En el servidor (con GPU):
  cd ~/Proyecto_V3
  source venv/bin/activate
  pip install Pillow

  # 3. Editar el dataset para usar rutas locales (las imágenes están en el servidor):
  #    O simplemente entrenar con el comando unificado:
  python fabrica_loras.py train \\
    --model Qwen/Qwen2-VL-2B-Instruct \\
    --data datasets/colores_vlm.jsonl \\
    --output adapters/colores_vlm/ \\
    --epochs 2

  # 4. Probar el adapter:
  python fabrica_loras.py chat \\
    --model adapters/colores_vlm/ \\
    --base-model Qwen/Qwen2-VL-2B-Instruct
""")

print(f"  {GREEN}✓{RESET} Dataset VLM listo. {BOLD}{n} ejemplos{RESET} en {jsonl_path}")
print(f"  Categorías: {list(categories.keys())}")
print(f"  Semáforo: {result.get('semaforo', '?')}")
