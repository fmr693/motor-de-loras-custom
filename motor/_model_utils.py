"""
motor._model_utils
==================
Helpers compartidos para carga de modelos con cuantización 4-bit NF4.

Uso:
    from motor._model_utils import apply_4bit_quantization

    model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    apply_4bit_quantization(model_kwargs, dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def apply_4bit_quantization(
    model_kwargs: Dict[str, Any],
    dtype: Any = None,
    *,
    cpu_offload: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Aplica configuración 4-bit NF4 a unos kwargs de HuggingFace.

    Modifica ``model_kwargs`` in-place añadiendo ``quantization_config``
    y eliminando ``torch_dtype`` (BitsAndBytesConfig lo gestiona).

    Parámetros
    ----------
    model_kwargs : dict
        Diccionario que se pasa a ``from_pretrained()``.
    dtype : torch.dtype
        Tipo de cómputo para 4-bit (ej: torch.float16).
    cpu_offload : bool
        Si True, activa ``llm_int8_enable_fp32_cpu_offload``
        para permitir offload a CPU cuando la VRAM es insuficiente.
    verbose : bool
        Si True, imprime el estado de la cuantización.

    Devuelve
    --------
    bool — True si la cuantización se aplicó correctamente.
    """
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        if verbose:
            print("  [WARN] bitsandbytes no instalado — cargando sin 4-bit")
            print("         pip install bitsandbytes para activar cuantización")
        return False

    bnb_kwargs: Dict[str, Any] = {
        "load_in_4bit":               True,
        "bnb_4bit_compute_dtype":     dtype,
        "bnb_4bit_use_double_quant":  True,
        "bnb_4bit_quant_type":        "nf4",
    }

    if cpu_offload:
        bnb_kwargs["llm_int8_enable_fp32_cpu_offload"] = True

    model_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)

    # torch_dtype lo gestiona BitsAndBytesConfig — quitarlo si existe
    model_kwargs.pop("torch_dtype", None)

    if verbose:
        msg = "[INFO] Cuantización 4-bit (NF4) activa"
        if cpu_offload:
            msg += " + CPU offload"
        print(f"  {msg}")

    return True


def get_device_info() -> Dict[str, Any]:
    """
    Detecta GPU y devuelve info del dispositivo.

    Devuelve
    --------
    dict con:
        device: "cuda" | "cpu"
        dtype: torch.dtype
        vram_gb: float (0 si CPU)
        gpu_name: str
        compute_capability: tuple | None
    """
    import torch

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0)
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        return {
            "device": "cuda",
            "dtype": dtype,
            "vram_gb": round(vram_gb, 1),
            "gpu_name": gpu_name,
            "compute_capability": cap,
        }
    else:
        return {
            "device": "cpu",
            "dtype": torch.float32,
            "vram_gb": 0.0,
            "gpu_name": "CPU",
            "compute_capability": None,
        }
