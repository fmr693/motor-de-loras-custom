"""
motor.exporter
==============
ExportManager: fusiona un adapter LoRA con su modelo base y exporta
el resultado a distintos formatos para distribución e inferencia.

Formatos soportados
-------------------
  safetensors  — Modelo fusionado en formato HuggingFace (default).
                 Compatible con transformers, vllm, text-generation-webui.
                 Solo requiere PEFT + transformers.

  gguf         — Formato comprimido para llama.cpp / Ollama / LM Studio.
                 Permite inferencia CPU sin GPU.
                 Requiere llama-cpp-python o el binario llama.cpp instalado.

Uso básico
----------
>>> from motor.exporter import ExportManager
>>> em = ExportManager(
...     adapter_dir = "adapters/titanic_llm",
...     base_model  = "Qwen/Qwen2.5-3B-Instruct",   # opcional, se lee del meta.json
... )
>>> em.to_safetensors("modelos/titanic_merged/")
>>> em.to_gguf("modelos/titanic_merged/", quantization="q4_k_m")

CLI (via fabrica_loras.py)
--------------------------
  python fabrica_loras.py export \\
      --adapter adapters/titanic_llm/ \\
      --output  modelos/titanic_merged/ \\
      --format  safetensors            # o gguf
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# ExportManager
# ---------------------------------------------------------------------------

class ExportManager:
    """
    Fusiona un adapter LoRA con su modelo base y exporta el resultado.

    Parámetros
    ----------
    adapter_dir : str
        Carpeta del adapter generada por LLMTrainer (contiene adapter_config.json,
        adapter_model.safetensors y meta.json).
    base_model : str | None
        ID del modelo base en HuggingFace. Si es None, se lee de meta.json.
    cache_dir : str | None
        Directorio de caché HuggingFace.
    """

    def __init__(
        self,
        adapter_dir : str,
        base_model  : Optional[str] = None,
        cache_dir   : Optional[str] = None,
    ):
        self.adapter_dir = Path(adapter_dir)
        self.cache_dir   = cache_dir

        # Leer meta.json para obtener modelo base si no se pasa explícito
        meta_path = self.adapter_dir / "meta.json"
        if base_model:
            self.base_model = base_model
        elif meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.base_model = meta.get("model_id") or meta.get("base_model")
            if not self.base_model:
                raise ValueError(
                    "meta.json no contiene 'model_id'. "
                    "Pasa --model explícitamente."
                )
        else:
            raise ValueError(
                f"No se encontró meta.json en {adapter_dir} y no se pasó --model."
            )

        print(f"[ExportManager] Adapter : {self.adapter_dir}")
        print(f"[ExportManager] Base    : {self.base_model}")

    # ------------------------------------------------------------------
    # Formato 1: safetensors (merge en HF)
    # ------------------------------------------------------------------

    def to_safetensors(self, output_dir: str) -> Path:
        """
        Fusiona adapter + base y guarda el modelo completo en safetensors.
        El resultado es un modelo HuggingFace estándar, listo para:
          - Cargarlo con AutoModelForCausalLM.from_pretrained(output_dir)
          - Subirlo a HuggingFace Hub
          - Usarlo con vllm, text-generation-webui, etc.

        No requiere GPU — el merge se hace en CPU si no hay CUDA disponible.
        """
        # Pre-cargar sklearn antes de transformers para evitar OSError [WinError 6714]
        # en Python 3.13 / Windows: _fill_cache() falla en directorios con TxF metadata
        # cuando el import ocurre dentro del lazy-loader de transformers.
        try:
            import sklearn, sklearn.externals, sklearn.externals._packaging  # noqa: F401
        except Exception:
            pass
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        print("\n[ExportManager] Cargando modelo base para merge...")
        # Estrategia de dispositivo para el merge:
        #   - GPU (VRAM): más rápido y evita OOM en servidores con poca RAM.
        #                 Usar si hay VRAM libre suficiente (>= 7 GB libres).
        #   - CPU (RAM):  fallback. Necesita tanta RAM como el modelo en fp16
        #                 (3B → ~7 GB, 7B → ~14 GB, 14B → ~28 GB).
        #   AVISO: asegúrate de parar el servidor FastAPI antes de exportar
        #          para liberar VRAM (kill -9 <PID> o Ctrl+C).
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0] / 1e9  # GB libres
            if free_vram >= 7:
                device = "cuda"
                dtype  = torch.float16
                print(f"       Usando GPU para el merge ({free_vram:.1f} GB VRAM libres)")
            else:
                device = "cpu"
                dtype  = torch.float16
                print(f"       VRAM insuficiente ({free_vram:.1f} GB libres). Usando CPU.")
                print(f"       ⚠ Si falla por OOM en RAM, para el servidor primero:")
                print(f"         kill -9 $(pgrep -f fabrica_loras)")
        else:
            device = "cpu"
            dtype  = torch.float16
            print("       Sin GPU detectada. Usando CPU.")

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            cache_dir         = self.cache_dir,
            trust_remote_code = True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype       = dtype,
            device_map        = device,
            cache_dir         = self.cache_dir,
            trust_remote_code = True,
        )

        print("[ExportManager] Cargando adapter LoRA...")
        model = PeftModel.from_pretrained(model, str(self.adapter_dir))

        print("[ExportManager] Fusionando adapter con modelo base (merge_and_unload)...")
        model = model.merge_and_unload()
        model.eval()

        print(f"[ExportManager] Guardando modelo fusionado en: {out}")
        model.save_pretrained(str(out), safe_serialization=True)
        tokenizer.save_pretrained(str(out))

        # Copiar meta.json al output para trazabilidad
        meta_src = self.adapter_dir / "meta.json"
        if meta_src.exists():
            shutil.copy(meta_src, out / "lora_meta.json")

        size_gb = sum(f.stat().st_size for f in out.rglob("*.safetensors")) / 1e9
        print(f"\n✓ Modelo fusionado guardado en: {out}")
        print(f"  Tamaño: {size_gb:.2f} GB")
        return out

    # ------------------------------------------------------------------
    # Formato 2: GGUF (para llama.cpp / Ollama / LM Studio)
    # ------------------------------------------------------------------

    def to_gguf(
        self,
        output_dir   : str,
        quantization : str = "q4_k_m",
        merged_dir   : Optional[str] = None,
    ) -> Path:
        """
        Exporta el modelo a formato GGUF para inferencia CPU con llama.cpp.

        Cuantizaciones disponibles (de menor a mayor calidad/tamaño):
          q2_k   → ~30% del tamaño original, calidad baja
          q4_k_m → ~45% del tamaño original, equilibrio recomendado  ← default
          q5_k_m → ~55% del tamaño original, buena calidad
          q8_0   → ~80% del tamaño original, casi sin pérdida
          f16    → 100% sin cuantización

        Requiere llama-cpp-python instalado:
          pip install llama-cpp-python

        Parámetros
        ----------
        output_dir   : carpeta de salida para el .gguf
        quantization : nivel de cuantización (default: q4_k_m)
        merged_dir   : si ya tienes el modelo fusionado en safetensors,
                       pásalo aquí para evitar rehacer el merge.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Paso 1: asegurar que tenemos el modelo fusionado en safetensors
        if merged_dir and Path(merged_dir).exists():
            merge_path = Path(merged_dir)
            print(f"[ExportManager] Usando modelo fusionado existente: {merge_path}")
        else:
            merge_path = out / "_merged_tmp"
            if merge_path.exists() and any(merge_path.iterdir()):
                print(f"[ExportManager] Reutilizando merge en caché: {merge_path}")
            else:
                print("[ExportManager] El formato GGUF requiere primero fusionar el adapter.")
                self.to_safetensors(str(merge_path))

        # Paso 2: convertir a GGUF via llama-cpp-python
        model_name = Path(self.base_model).name if "/" not in self.base_model \
                     else self.base_model.split("/")[-1]
        gguf_path  = out / f"{model_name}-{quantization}.gguf"

        print(f"\n[ExportManager] Convirtiendo a GGUF ({quantization})...")

        try:
            self._convert_via_llama_cpp(merge_path, gguf_path, quantization)
        except FileNotFoundError:
            raise RuntimeError(
                "llama-cpp-python no está instalado o no se encontró el script convert.\n"
                "Instálalo con: pip install llama-cpp-python\n"
                "O usa --format safetensors para exportar sin GGUF."
            )

        # Limpiar merge temporal si lo creamos nosotros
        if not merged_dir and (out / "_merged_tmp").exists():
            shutil.rmtree(out / "_merged_tmp")
            print("[ExportManager] Carpeta temporal de merge eliminada.")

        size_mb = gguf_path.stat().st_size / 1e6
        print(f"\n✓ GGUF guardado en: {gguf_path}")
        print(f"  Tamaño: {size_mb:.0f} MB")
        return gguf_path

    def _convert_via_llama_cpp(self, model_dir: Path, gguf_path: Path, quant: str):
        """
        Convierte un modelo HuggingFace a GGUF usando los scripts de llama.cpp.

        La estrategia es:
          1. Buscar convert_hf_to_gguf.py en rutas conocidas (incluyendo dentro del
             paquete llama_cpp, que en algunas versiones lo incluye).
          2. Si no se encuentra, clonar llama.cpp en ~/llama.cpp (una sola vez, ~200 MB).
          3. Convertir a fp16 GGUF.
          4. Cuantizar con llama-quantize si está disponible; si no, guardar en f16.

        Nota: llama-cpp-python NO incluye convert_hf_to_gguf.py en versiones modernas.
        El script pertenece al repositorio llama.cpp (C++), no al wrapper Python.
        """
        # ── Paso 0: localizar convert_hf_to_gguf.py ──────────────────────────────
        try:
            import llama_cpp
            pkg_dir = Path(llama_cpp.__file__).parent
        except ImportError:
            pkg_dir = None

        candidates: list[Path] = []
        if pkg_dir:
            # En algunas builds el script está directamente en el paquete Python
            candidates += [
                pkg_dir / "convert_hf_to_gguf.py",
                pkg_dir.parent / "convert_hf_to_gguf.py",
            ]
            candidates += list(pkg_dir.rglob("convert_hf_to_gguf.py"))

        candidates += [
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path("/tmp/llama.cpp/convert_hf_to_gguf.py"),
            Path.cwd() / "llama.cpp" / "convert_hf_to_gguf.py",
        ]

        convert_py = next((p for p in candidates if p.exists()), None)

        if not convert_py:
            # Clonar llama.cpp desde GitHub (repositorio oficial, solo lectura, depth=1)
            clone_dir = Path.home() / "llama.cpp"
            print("[ExportManager] convert_hf_to_gguf.py no encontrado.")
            print("[ExportManager] Clonando llama.cpp (una sola vez, ~200 MB)...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ggerganov/llama.cpp", str(clone_dir)],
                check=True,
            )
            # Instalar solo los paquetes que necesita convert_hf_to_gguf.py.
            # No usamos requirements.txt porque pina numpy~=1.26.4 que no
            # tiene wheel pre-compilada para Python 3.13 y falla sin compilador C.
            print("[ExportManager] Instalando dependencias del script de conversión...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "sentencepiece>=0.1.98", "gguf>=0.1.0",
                 "protobuf>=4.21.0,<5.0.0", "-q"],
                check=True,
            )
            convert_py = clone_dir / "convert_hf_to_gguf.py"

        if not convert_py.exists():
            raise FileNotFoundError(
                "No se pudo localizar ni clonar convert_hf_to_gguf.py.\n"
                "Clónalo manualmente y vuelve a intentarlo:\n"
                "  git clone --depth 1 https://github.com/ggerganov/llama.cpp ~/llama.cpp"
            )

        # ── Paso 1: HuggingFace → GGUF fp16 ──────────────────────────────────────
        fp16_path = gguf_path.parent / f"{gguf_path.stem}-fp16.gguf"
        cmd_convert = [
            sys.executable, str(convert_py),
            str(model_dir),
            "--outfile", str(fp16_path),
            "--outtype", "f16",
        ]
        print(f"  [1/2] Convirtiendo HF → GGUF fp16...")
        print(f"        {' '.join(cmd_convert)}")
        subprocess.run(cmd_convert, check=True)

        if quant == "f16":
            fp16_path.rename(gguf_path)
            return

        # ── Paso 2: Cuantizar fp16 → q4_k_m (o el nivel solicitado) ─────────────
        quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
        if not quantize_bin:
            # Buscar el binario compilado en el repo clonado
            built = convert_py.parent / "build" / "bin" / "llama-quantize"
            if built.exists():
                quantize_bin = str(built)

        if quantize_bin:
            cmd_quant = [quantize_bin, str(fp16_path), str(gguf_path), quant.upper()]
            print(f"  [2/2] Cuantizando a {quant}...")
            print(f"        {' '.join(cmd_quant)}")
            subprocess.run(cmd_quant, check=True)
            fp16_path.unlink(missing_ok=True)
        else:
            print(f"  [WARN] llama-quantize no encontrado. Guardando en f16 (más grande).")
            print(f"         Para compilarlo en el servidor:")
            print(f"           cd ~/llama.cpp && cmake -B build && cmake --build build -j")
            fp16_path.rename(gguf_path)

    # ------------------------------------------------------------------
    # Info del adapter
    # ------------------------------------------------------------------

    def info(self) -> dict:
        """Devuelve metadatos del adapter y estima el tamaño del modelo fusionado."""
        meta_path = self.adapter_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        # Estimar tamaño de los safetensors del adapter
        adapter_mb = sum(
            f.stat().st_size for f in self.adapter_dir.rglob("*.safetensors")
        ) / 1e6

        return {
            "adapter_dir"  : str(self.adapter_dir),
            "base_model"   : self.base_model,
            "adapter_mb"   : round(adapter_mb, 1),
            "meta"         : meta,
        }
