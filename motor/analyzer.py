"""
motor.analyzer
==============
ModelAnalyzer: dada la ID de un modelo HuggingFace, determina automáticamente
todo lo necesario para entrenarlo con LoRA sin descargar sus pesos.

Qué hace:
  1. Lee config.json y tokenizer_config.json del Hub (solo ~10 KB por modelo)
  2. Detecta familia del modelo (qwen2, llama, mistral, gemma, …)
  3. Detecta si es un VLM (visión + lenguaje)
  4. Mapea los target_modules correctos para LoRA según la familia
  5. Detecta el chat template del tokenizador
  6. Selecciona el motor de entrenamiento óptimo (unsloth / llamafactory / peft_trl)
  7. Estima la VRAM mínima necesaria

Uso básico
----------
>>> from motor.analyzer import ModelAnalyzer
>>> a = ModelAnalyzer("Qwen/Qwen2.5-7B-Instruct")
>>> cfg = a.analyze()
>>> print(cfg)
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "family": "qwen2",
  "is_vlm": False,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", ...],
  "chat_template": "chatml",
  "engine": "unsloth",
  "vram_estimate_gb": {"fp16": 14.0, "4bit": 4.0},
  "lora_config": {...},
}

Funciona sin conexión si el modelo ya está en caché local de HuggingFace.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ===========================================================================
# Mapa de familias conocidas
# ===========================================================================

# model_type (del config.json de HF) → metadatos de la familia
_FAMILY_MAP: Dict[str, Dict[str, Any]] = {
    # ---- Qwen2 / Qwen2.5 texto ----
    "qwen2": {
        "family":         "qwen2",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "chat_template":  "chatml",
        "engine":         "unsloth",
        "unsloth_id":     "qwen2",   # para FastLanguageModel.from_pretrained
    },
    # ---- Qwen2-VL (multimodal) ----
    "qwen2_vl": {
        "family":         "qwen2_vl",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "chat_template":  "chatml",
        "engine":         "llamafactory",
        "is_vlm":         True,
    },
    # ---- LLaMA 2 / 3 / 3.1 / 3.2 ----
    "llama": {
        "family":         "llama",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "chat_template":  "llama3",
        "engine":         "unsloth",
        "unsloth_id":     "llama",
    },
    # ---- Mistral / Mixtral ----
    "mistral": {
        "family":         "mistral",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "chat_template":  "chatml",
        "engine":         "unsloth",
        "unsloth_id":     "mistral",
    },
    "mixtral": {
        "family":         "mistral",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "chat_template":  "chatml",
        "engine":         "unsloth",
        "unsloth_id":     "mistral",
    },
    # ---- Gemma / Gemma2 ----
    "gemma": {
        "family":         "gemma",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "chat_template":  "gemma",
        "engine":         "unsloth",
        "unsloth_id":     "gemma",
    },
    "gemma2": {
        "family":         "gemma2",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "chat_template":  "gemma",
        "engine":         "unsloth",
        "unsloth_id":     "gemma",
    },
    "gemma3": {
        "family":         "gemma3",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "chat_template":  "gemma",
        "engine":         "peft_trl",
    },
    # ---- Gemma 4 Unified (encoder-free, multimodal: texto+imagen+audio) ----
    # Se entrena como texto "in one pass" (model card oficial), así que
    # is_vlm=False explícito: tiene vision_config/audio_config pero NO debe
    # ir al pipeline VLM. Carga con AutoProcessor + AutoModelForMultimodalLM
    # (multimodal_lm=True), requiere transformers reciente (> 5.5).
    # Las capas globales usan KV unificado (attention_k_eq_v) → el trainer
    # filtra los target_modules contra los módulos reales del modelo.
    "gemma4_unified": {
        "family":         "gemma4",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "chat_template":  "gemma",
        "engine":         "peft_trl",
        "is_vlm":         False,
        "multimodal_lm":  True,
    },
    # ---- Phi-3 ----
    "phi3": {
        "family":         "phi3",
        "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "chat_template":  "phi3",
        "engine":         "unsloth",
        "unsloth_id":     "phi3",
    },
    # ---- CLIP (visión-lenguaje clásico) ----
    "clip": {
        "family":         "clip",
        "target_modules": ["q_proj", "v_proj"],
        "chat_template":  None,
        "engine":         "peft_trl",
        "is_vlm":         True,
    },
    # ---- XLM-RoBERTa / BERT family ----
    "xlm-roberta": {
        "family":         "xlm_roberta",
        "target_modules": ["query", "value"],
        "chat_template":  None,
        "engine":         "peft_trl",
    },
    "roberta": {
        "family":         "roberta",
        "target_modules": ["query", "value"],
        "chat_template":  None,
        "engine":         "peft_trl",
    },
    "bert": {
        "family":         "bert",
        "target_modules": ["query", "value"],
        "chat_template":  None,
        "engine":         "peft_trl",
    },
}

# Fallback genérico cuando la familia no está en el mapa
_GENERIC_FAMILY = {
    "family":         "generic",
    "target_modules": ["q_proj", "v_proj"],
    "chat_template":  "chatml",
    "engine":         "peft_trl",
}

# Motor → descripción legible
_ENGINE_DESCRIPTION = {
    "unsloth":      "Unsloth (2× velocidad, 70% menos VRAM, recomendado para LLMs texto)",
    "llamafactory": "LLaMA-Factory (soporte VLM multimodal: Qwen2-VL, LLaVA, InternVL)",
    "peft_trl":     "PEFT + TRL (arquitecturas custom: CLIP, XLM-R, BERT; máximo control)",
}

# ===========================================================================
# Perfiles de GPU (S3.1)
# ===========================================================================

# Fuente: Tim Dettmers (2023), NVIDIA specs, benchmarks comunitarios
# Factor clave: memory bandwidth determina ~70% de la velocidad de entrenamiento

GPU_PROFILES: Dict[str, Dict[str, Any]] = {
    "RTX 4090": {
        "vram_gb": 24, "bandwidth_gbs": 1008, "fp16_tflops": 82.6,
        "arch": "Ada Lovelace", "compute_capability": "8.9",
        "supports_bf16": True, "supports_fp8": True,
    },
    "RTX 4080": {
        "vram_gb": 17.2, "bandwidth_gbs": 717, "fp16_tflops": 48.7,
        "arch": "Ada Lovelace", "compute_capability": "8.9",
        "supports_bf16": True, "supports_fp8": True,
    },
    "RTX 4070 Ti": {
        "vram_gb": 12, "bandwidth_gbs": 504, "fp16_tflops": 40.1,
        "arch": "Ada Lovelace", "compute_capability": "8.9",
        "supports_bf16": True, "supports_fp8": True,
    },
    "RTX 3090": {
        "vram_gb": 24, "bandwidth_gbs": 936, "fp16_tflops": 35.6,
        "arch": "Ampere", "compute_capability": "8.6",
        "supports_bf16": True, "supports_fp8": False,
    },
    "RTX 3080": {
        "vram_gb": 10, "bandwidth_gbs": 760, "fp16_tflops": 29.8,
        "arch": "Ampere", "compute_capability": "8.6",
        "supports_bf16": True, "supports_fp8": False,
    },
    "RTX 3070": {
        "vram_gb": 8, "bandwidth_gbs": 448, "fp16_tflops": 20.3,
        "arch": "Ampere", "compute_capability": "8.6",
        "supports_bf16": True, "supports_fp8": False,
    },
    "GTX 1080 Ti": {
        "vram_gb": 11, "bandwidth_gbs": 484, "fp16_tflops": 0.17,
        "arch": "Pascal", "compute_capability": "6.1",
        "supports_bf16": False, "supports_fp8": False,
    },
    "Tesla T4": {
        "vram_gb": 16, "bandwidth_gbs": 320, "fp16_tflops": 8.1,
        "arch": "Turing", "compute_capability": "7.5",
        "supports_bf16": False, "supports_fp8": False,
    },
    "A100": {
        "vram_gb": 40, "bandwidth_gbs": 1555, "fp16_tflops": 312,
        "arch": "Ampere", "compute_capability": "8.0",
        "supports_bf16": True, "supports_fp8": False,
    },
    "H100": {
        "vram_gb": 80, "bandwidth_gbs": 3350, "fp16_tflops": 990,
        "arch": "Hopper", "compute_capability": "9.0",
        "supports_bf16": True, "supports_fp8": True,
    },
}

# Factor de velocidad relativa para CPU-only (sin GPU)
CPU_BANDWIDTH_GBS = 50.0  # DDR5-5600 dual channel ~50 GB/s
CPU_SPEED_FACTOR = 0.07   # ~10-20× más lento que GPU con Tensor Cores

# Baselines para extrapolación de tiempos
# (datos empíricos del proyecto lora-factory)
_BENCHMARKS = [
    {   # Titanic 3B en GTX 1080 Ti
        "model_params_b": 3.0, "num_examples": 891, "avg_tokens": 85,
        "epochs": 3, "gpu": "GTX 1080 Ti", "minutes": 31,
    },
    {   # Finance 14B en RTX 4080
        "model_params_b": 14.0, "num_examples": 11931, "avg_tokens": 62,
        "epochs": 3, "gpu": "RTX 4080", "minutes": 210,
    },
]


# ===========================================================================
# Resultado del análisis
# ===========================================================================

@dataclass
class ModelAnalysisResult:
    """Resultado completo del análisis de un modelo."""
    model_id:        str
    family:          str
    model_type:      str                      # valor crudo de config.json
    is_vlm:          bool
    num_params_b:    Optional[float]          # parámetros en billones (aprox.)
    target_modules:  List[str]
    chat_template:   Optional[str]
    engine:          str
    engine_desc:     str
    vram_estimate:   Dict[str, float]         # {"fp16": X, "8bit": X, "4bit": X}
    lora_config:     Dict[str, Any]           # listo para LoraConfig(**lora_config)
    warnings:        List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def print_summary(self) -> None:
        """Imprime un resumen legible del análisis."""
        sep = "=" * 60
        print(sep)
        print(f"  ModelAnalyzer — {self.model_id}")
        print(sep)
        print(f"  Familia:          {self.family}")
        print(f"  Tipo raw:         {self.model_type}")
        print(f"  ¿Es VLM?:         {'Sí' if self.is_vlm else 'No'}")
        if self.num_params_b:
            print(f"  Parámetros:       ~{self.num_params_b:.1f}B")
        print(f"  Target modules:   {self.target_modules}")
        print(f"  Chat template:    {self.chat_template or 'N/A'}")
        print(f"  Motor sugerido:   {self.engine_desc}")
        print()
        print("  Estimación VRAM mínima:")
        for prec, gb in self.vram_estimate.items():
            print(f"    {prec:8s}: {gb:.1f} GB")
        print()
        print("  LoRA config sugerida:")
        for k, v in self.lora_config.items():
            print(f"    {k:25s}: {v}")
        if self.warnings:
            print()
            print("  ⚠  Advertencias:")
            for w in self.warnings:
                # Separar visualmente las recomendaciones de tamaño
                if w.startswith("TAMAÑO") or w.startswith("MODELO"):
                    print(f"    💡 {w}")
                else:
                    print(f"    - {w}")
        print(sep)


# ===========================================================================
# ModelAnalyzer
# ===========================================================================

class ModelAnalyzer:
    """
    Analiza un modelo HuggingFace sin descargar sus pesos.

    Parámetros
    ----------
    model_id : str
        ID del modelo en HuggingFace Hub. Ej: "Qwen/Qwen2.5-7B-Instruct"
        También acepta rutas locales si el modelo ya está descargado.
    cache_dir : str, opcional
        Directorio de caché de HuggingFace. Si es None usa el por defecto
        (~/.cache/huggingface/hub).
    lora_r : int
        Rango LoRA sugerido. Por defecto 16.
    lora_alpha : int
        Alpha LoRA sugerido. Por defecto 32 (convención: alpha = 2 * r).
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        self.model_id  = model_id.strip()
        self.cache_dir = cache_dir
        self.lora_r    = lora_r
        self.lora_alpha = lora_alpha

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def analyze(self) -> ModelAnalysisResult:
        """
        Ejecuta el análisis completo y devuelve un ModelAnalysisResult.
        Lanza excepciones con mensajes claros si el modelo no es accesible.
        """
        warnings: List[str] = []

        print(f"[ModelAnalyzer] Analizando: {self.model_id}")

        # 1. Leer config.json
        config = self._fetch_json("config.json")
        if config is None:
            raise RuntimeError(
                f"[ModelAnalyzer] No se pudo leer config.json de '{self.model_id}'.\n"
                "Verifica que el model_id es correcto y tienes conexión a internet,\n"
                "o que el modelo está descargado en caché local."
            )

        model_type = config.get("model_type", "unknown").lower()
        print(f"  model_type detectado: {model_type}")

        # 2. Buscar familia en el mapa
        family_info = _FAMILY_MAP.get(model_type)
        if family_info is None:
            warnings.append(
                f"Familia '{model_type}' no reconocida — usando configuración genérica. "
                "Revisa manualmente los target_modules antes de entrenar."
            )
            family_info = dict(_GENERIC_FAMILY)
        else:
            family_info = dict(family_info)

        # 3. Detectar VLM (tiene vision_config o el model_type ya lo indica).
        # Si la familia lo declara explícitamente, manda sobre la heurística:
        # gemma4_unified tiene vision_config pero se entrena como texto.
        _explicit_vlm = family_info.get("is_vlm")
        if _explicit_vlm is not None:
            is_vlm = bool(_explicit_vlm)
        else:
            is_vlm = bool(
                config.get("vision_config") is not None
                or "vision" in model_type
            )
        if is_vlm and family_info.get("engine") == "unsloth":
            # Si es VLM pero Unsloth no lo soporta, cambiar a llamafactory
            family_info["engine"] = "llamafactory"
            warnings.append(
                "Modelo con visión detectado — motor cambiado a LLaMA-Factory "
                "(Unsloth no soporta VLMs)."
            )

        # 4. Estimar número de parámetros
        num_params_b = self._estimate_params(config)
        self._last_num_params = num_params_b  # cache para recommend()

        # 5. Chat template — leer tokenizer_config.json
        chat_template = family_info.get("chat_template")
        tok_config = self._fetch_json("tokenizer_config.json")
        if tok_config:
            raw_template = tok_config.get("chat_template")
            if raw_template:
                detected = self._classify_chat_template(raw_template)
                if detected and detected != chat_template:
                    chat_template = detected
                    print(f"  Chat template detectado en tokenizer_config: {chat_template}")

        # 6. VRAM estimada
        vram = self._estimate_vram(num_params_b)

        # 7. Advertencias adicionales
        if num_params_b and num_params_b > 13:
            warnings.append(
                f"Modelo grande (~{num_params_b:.0f}B params). "
                "Requiere cuantización 4-bit para entrenar en ≤ 24 GB VRAM."
            )
        if family_info["engine"] == "llamafactory":
            warnings.append(
                "LLaMA-Factory requiere instalación separada: "
                "pip install llamafactory"
            )
        if model_type == "unknown":
            warnings.append(
                "model_type='unknown' — el config.json no declara el tipo de modelo."
            )

        # 7b. Recomendador de tamaño de modelo
        size_warnings = self._check_model_size(num_params_b)
        warnings.extend(size_warnings)

        # 8. LoRA config sugerida
        lora_config = self._build_lora_config(family_info, is_vlm)

        engine = family_info["engine"]
        result = ModelAnalysisResult(
            model_id       = self.model_id,
            family         = family_info["family"],
            model_type     = model_type,
            is_vlm         = is_vlm,
            num_params_b   = num_params_b,
            target_modules = family_info["target_modules"],
            chat_template  = chat_template,
            engine         = engine,
            engine_desc    = _ENGINE_DESCRIPTION.get(engine, engine),
            vram_estimate  = vram,
            lora_config    = lora_config,
            warnings       = warnings,
        )

        result.print_summary()
        return result

    # ------------------------------------------------------------------
    # Helpers de lectura de archivos del Hub
    # ------------------------------------------------------------------

    def _fetch_json(self, filename: str) -> Optional[dict]:
        """
        Intenta leer un JSON del modelo desde HuggingFace Hub.
        Primero prueba caché local; si no, descarga solo el fichero (sin pesos).
        """
        # Opción 1: ruta local absoluta o relativa
        local = Path(self.model_id) / filename
        if local.exists():
            try:
                return json.loads(local.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Opción 2: caché de HuggingFace Hub (ya descargado antes)
        cached = self._find_in_hf_cache(filename)
        if cached:
            try:
                return json.loads(Path(cached).read_text(encoding="utf-8"))
            except Exception:
                pass

        # Opción 3: descargar solo el fichero JSON desde el Hub
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id   = self.model_id,
                filename  = filename,
                cache_dir = self.cache_dir,
            )
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except ImportError:
            pass  # huggingface_hub no instalado, probar con requests
        except Exception as e:
            print(f"  [WARN] hf_hub_download falló para {filename}: {e}")

        # Opción 4: petición HTTP directa a raw.githubusercontent / hf.co
        return self._fetch_json_http(filename)

    def _fetch_json_http(self, filename: str) -> Optional[dict]:
        """Descarga un JSON directamente desde la API de HuggingFace."""
        import urllib.request
        import urllib.error

        # Formato URL del Hub
        url = f"https://huggingface.co/{self.model_id}/resolve/main/{filename}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "motor-de-loras/0.1"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print(f"  [WARN] Modelo privado — autenticación requerida para {filename}")
            elif e.code == 404:
                print(f"  [WARN] {filename} no encontrado en el Hub para {self.model_id}")
            else:
                print(f"  [WARN] HTTP {e.code} al descargar {filename}")
        except Exception as e:
            print(f"  [WARN] Error de red descargando {filename}: {e}")
        return None

    def _find_in_hf_cache(self, filename: str) -> Optional[str]:
        """Busca un fichero en la caché local de HuggingFace Hub."""
        hf_cache = Path(
            self.cache_dir
            or os.environ.get("HF_HOME", "")
            or Path.home() / ".cache" / "huggingface" / "hub"
        )
        if not hf_cache.exists():
            return None

        # Formato de carpeta en caché: models--ORG--NOMBRE/snapshots/HASH/filename
        model_folder_name = "models--" + self.model_id.replace("/", "--")
        model_dir = hf_cache / model_folder_name

        if not model_dir.exists():
            return None

        # Buscar en todos los snapshots disponibles
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        for snapshot in sorted(snapshots_dir.iterdir(), reverse=True):
            candidate = snapshot / filename
            if candidate.exists():
                return str(candidate)
        return None

    # ------------------------------------------------------------------
    # Helpers de análisis
    # ------------------------------------------------------------------

    def _estimate_params(self, config: dict) -> Optional[float]:
        """
        Estima el número de parámetros en billones a partir de config.json.
        Usa num_parameters si está disponible, sino estima por arquitectura.
        """
        # Algunos modelos declaran el número exacto
        if "num_parameters" in config:
            return config["num_parameters"] / 1e9

        # Estimación por arquitectura transformer estándar
        hidden = config.get("hidden_size")
        layers = config.get("num_hidden_layers")
        vocab  = config.get("vocab_size")
        intermediate = config.get("intermediate_size") or (hidden * 4 if hidden else None)

        if not (hidden and layers):
            return None

        # Aprox: embedding + capas attention + capas FFN
        embed_params = (vocab or 32000) * hidden
        attn_params  = layers * 4 * hidden * hidden          # Q, K, V, O
        ffn_params   = layers * 3 * hidden * (intermediate or hidden * 4)  # gate, up, down
        total = (embed_params + attn_params + ffn_params) / 1e9
        return round(total, 1)

    def _estimate_vram(self, num_params_b: Optional[float]) -> Dict[str, float]:
        """
        Estima VRAM mínima en GB para distintas precisiones.
        Incluye overhead de activaciones y gradientes (~20%).
        """
        if not num_params_b:
            return {"fp16": 0.0, "8bit": 0.0, "4bit": 0.0}

        # Bytes por parámetro según precisión
        overhead = 1.2   # 20% activaciones + estados del optimizador (LoRA entrena poco)
        fp16 = round(num_params_b * 2 * overhead, 1)
        bit8 = round(num_params_b * 1 * overhead, 1)
        bit4 = round(num_params_b * 0.5 * overhead, 1)
        return {"fp16": fp16, "8bit": bit8, "4bit": bit4}

    def _check_model_size(self, num_params_b: Optional[float]) -> List[str]:
        """
        Evalúa el tamaño del modelo y genera recomendaciones basadas en
        la experiencia de uso real con adapters LoRA.

        Umbrales basados en observaciones:
          < 1B  → demasiado pequeño para tareas complejas
          1-3B  → válido solo para tareas muy específicas (clasificación, YES/NO)
          3-7B  → zona intermedia, limitaciones en razonamiento multipasos
          7-13B → equilibrio calidad/recursos, recomendado para uso general
          > 13B → calidad alta, requiere cuantización 4-bit en GPUs ≤ 24GB
        """
        if not num_params_b:
            return []

        msgs = []

        if num_params_b < 1.0:
            msgs.append(
                f"TAMAÑO INSUFICIENTE (~{num_params_b:.1f}B): modelo demasiado pequeño. "
                "Capacidad de razonamiento muy limitada. "
                "Recomendado mínimo: 1B para clasificación, 7B para uso general."
            )
        elif num_params_b < 3.0:
            msgs.append(
                f"MODELO PEQUEÑO (~{num_params_b:.1f}B): adecuado solo para tareas "
                "muy específicas (clasificación binaria, extracción de campos fijos). "
                "Limitaciones: puede ignorar preguntas múltiples, respuestas cortas, "
                "historial de conversación reducido. Para uso general usar ≥ 7B."
            )
        elif num_params_b < 7.0:
            msgs.append(
                f"MODELO MEDIANO (~{num_params_b:.1f}B): funcional para tareas "
                "específicas pero con limitaciones en razonamiento complejo y "
                "preguntas heterogéneas. Para conversación general o tareas mixtas "
                "se recomienda ≥ 7B."
            )
        elif num_params_b <= 13.0:
            msgs.append(
                f"TAMAÑO ÓPTIMO (~{num_params_b:.1f}B): equilibrio calidad/recursos "
                "recomendado para la mayoría de casos de uso con LoRA. "
                "Compatible con GPUs de 16-24 GB VRAM en 4-bit."
            )
        # > 13B ya está cubierto por el aviso de VRAM en el bloque anterior

        return msgs

    # ------------------------------------------------------------------
    # Recomendaciones inteligentes (S3.1)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_gpu() -> Optional[str]:
        """
        Detecta la GPU instalada usando torch/tensorflow o nvidia-smi.

        Devuelve el nombre de la GPU (ej: "RTX 4080") o None si no hay GPU.
        """
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                # Normalizar nombres comunes
                for gpu_name in GPU_PROFILES:
                    if gpu_name.lower() in name.lower():
                        return gpu_name
                return name
        except ImportError:
            pass

        # Fallback: nvidia-smi
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            name = result.stdout.strip().split("\n")[0]
            for gpu_name in GPU_PROFILES:
                if gpu_name.lower() in name.lower():
                    return gpu_name
            return name
        except Exception:
            return None

    def recommend(
        self,
        num_examples: int,
        avg_tokens: int = 100,
        num_classes: int = 2,
        gpu: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recomienda hiperparámetros LoRA basados en el dataset y modelo.

        Usa reglas derivadas de la comunidad LoRA, papers (LIMA, QLoRA)
        y benchmarks empíricos del proyecto.

        Parámetros
        ----------
        num_examples : int
            Número de ejemplos en el dataset.
        avg_tokens : int
            Longitud media de los ejemplos en tokens.
        num_classes : int
            Número de clases distintas (1 = modo extracción/generación).
        gpu : str, opcional
            Nombre de la GPU objetivo. Si None, auto-detecta.

        Devuelve
        --------
        dict con claves:
            lora_r, lora_alpha, epochs, learning_rate,
            needs_augmentation, reasoning, time_estimate, vram_estimate
        """
        # ── Detectar GPU ─────────────────────────────────────────────
        gpu = gpu or self._detect_gpu()
        gpu_profile = GPU_PROFILES.get(gpu, {}) if gpu else {}
        is_cpu = gpu is None

        # ── Hiperparámetros ──────────────────────────────────────────
        if num_examples < 200:
            r, alpha = 4, 8
            epochs = 5
        elif num_examples < 500:
            r, alpha = 8, 16
            epochs = 4
        elif num_examples < 2000:
            r, alpha = 16, 32
            epochs = 3
        elif num_examples < 10000:
            r, alpha = 32, 64
            epochs = 2
        else:
            r, alpha = 32, 64
            epochs = 1

        lr = 5e-5 if r <= 8 else 2e-4

        # ── ¿Necesita augmentation? ──────────────────────────────────
        needs_aug = (
            num_examples < 200
            or (num_classes > 1 and num_examples / max(num_classes, 1) < 20)
        )

        # ── Justificación ────────────────────────────────────────────
        reasons = []
        reasons.append(
            f"r={r}: dataset {'muy pequeño' if num_examples < 500 else 'mediano' if num_examples < 2000 else 'grande'}"
        )
        reasons.append(f"epochs={epochs}: {'pocos datos → más épocas' if epochs > 2 else 'datos suficientes → menos épocas'}")
        reasons.append(f"alpha={alpha} (2× r, convención estándar)")
        if num_classes > 10:
            reasons.append(f"{num_classes} clases → r moderado para evitar overfitting por clase")

        # ── Tiempo estimado ──────────────────────────────────────────
        time_est = self.estimate_time(
            num_examples=num_examples,
            avg_tokens=avg_tokens,
            epochs=epochs,
            gpu=gpu,
        )

        # ── VRAM estimada ────────────────────────────────────────────
        vram_est = self._estimate_vram(
            self._last_num_params if hasattr(self, '_last_num_params') else None
        )

        return {
            "lora_r": r,
            "lora_alpha": alpha,
            "epochs": epochs,
            "learning_rate": lr,
            "needs_augmentation": needs_aug,
            "reasoning": " | ".join(reasons),
            "time_estimate": time_est,
            "vram_estimate_4bit": vram_est.get("4bit", 0) if vram_est else 0,
            "gpu": gpu or "CPU",
            "gpu_bandwidth_gbs": gpu_profile.get("bandwidth_gbs", CPU_BANDWIDTH_GBS),
            "gpu_vram_gb": gpu_profile.get("vram_gb", 0),
            "warnings": self._generate_warnings(
                num_examples, num_classes, r, epochs,
                gpu_profile.get("vram_gb", 0),
                vram_est.get("4bit", 0) if vram_est else 0,
                is_cpu,
            ),
        }

    def estimate_time(
        self,
        num_examples: int,
        avg_tokens: int = 100,
        epochs: int = 3,
        gpu: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Estima el tiempo de entrenamiento basado en física de GPU
        (memory bandwidth) + benchmarks empíricos del proyecto.

        Parámetros
        ----------
        num_examples : int
        avg_tokens : int
        epochs : int
        gpu : str, opcional

        Devuelve
        --------
        dict con: estimated_minutes, confidence, formula_used, benchmarks_used
        """
        gpu = gpu or self._detect_gpu()
        is_cpu = gpu is None
        gpu_profile = GPU_PROFILES.get(gpu, {}) if gpu else {}

        # ── Estimar params del modelo ────────────────────────────────
        model_params_b = self._last_num_params if hasattr(self, '_last_num_params') else 7.0

        # Factor de GPU (relativo a RTX 4080 baseline)
        baseline_bw = 717.0  # RTX 4080
        if is_cpu:
            bw_factor = CPU_SPEED_FACTOR
        else:
            gpu_bw = gpu_profile.get("bandwidth_gbs", baseline_bw)
            bw_factor = gpu_bw / baseline_bw

        # ── Fórmula basada en física ─────────────────────────────────
        # Tiempo ∝ (params × tokens × epochs) / bandwidth
        total_tokens = num_examples * avg_tokens * epochs
        # Constante empírica calibrada con benchmark Titanic (3B, 891, 85tok, 3ep, 31min)
        TOKENS_PER_MINUTE_BASELINE = 45000  # tokens/min para 1B params en RTX 4080
        estimated_minutes = (
            total_tokens * (model_params_b / 1.0) / TOKENS_PER_MINUTE_BASELINE
        ) / max(bw_factor, 0.01)

        # ── Confianza ────────────────────────────────────────────────
        # Alta si tenemos un benchmark cercano, media si extrapolamos
        confidence = "baja"
        for bench in _BENCHMARKS:
            params_diff = abs(bench["model_params_b"] - model_params_b) / max(model_params_b, 1)
            tokens_diff = abs(bench["num_examples"] - num_examples) / max(num_examples, 1)
            if params_diff < 0.5 and tokens_diff < 2.0:
                confidence = "alta"
                break
            elif params_diff < 1.5 and tokens_diff < 5.0:
                confidence = "media"

        # Encontrar benchmark más cercano para referencia
        closest_bench = min(
            _BENCHMARKS,
            key=lambda b: abs(b["model_params_b"] - model_params_b)
        )

        # ── Rango ────────────────────────────────────────────────────
        min_est = round(estimated_minutes * 0.7)
        max_est = round(estimated_minutes * 1.5)

        return {
            "estimated_minutes": round(estimated_minutes),
            "range_min": min_est,
            "range_max": max_est,
            "confidence": confidence,
            "total_tokens": total_tokens,
            "bw_factor": round(bw_factor, 2),
            "closest_benchmark": f"{closest_bench['model_params_b']}B model, "
                                 f"{closest_bench['num_examples']} examples, "
                                 f"{closest_bench['minutes']} min on {closest_bench['gpu']}",
        }

    @staticmethod
    def _generate_warnings(
        num_examples: int,
        num_classes: int,
        r: int,
        epochs: int,
        gpu_vram_gb: float,
        vram_needed_4bit: float,
        is_cpu: bool,
    ) -> List[str]:
        """Genera advertencias basadas en la configuración recomendada."""
        warnings = []

        if is_cpu:
            warnings.append(
                "CPU-only detectado. El entrenamiento sera 10-20x mas lento. "
                "Recomendado solo para prototipado con modelos <3B."
            )
        elif gpu_vram_gb > 0 and vram_needed_4bit > gpu_vram_gb * 0.9:
            warnings.append(
                f"VRAM ajustada: el modelo necesita ~{vram_needed_4bit:.1f} GB en 4-bit, "
                f"tu GPU tiene {gpu_vram_gb} GB. Puede que no quepa o necesites batch_size=1."
            )
            # Sugerir multi-GPU si la VRAM es muy insuficiente
            if vram_needed_4bit > gpu_vram_gb * 1.5:
                import torch as _t
                try:
                    n_gpus = _t.cuda.device_count() if _t.cuda.is_available() else 0
                except Exception:
                    n_gpus = 0
                if n_gpus <= 1:
                    warnings.append(
                        f"El modelo requiere {vram_needed_4bit:.1f} GB en 4-bit pero "
                        f"solo tienes {gpu_vram_gb} GB. Recomendado: usar un modelo "
                        f"mas pequeño o añadir GPUs adicionales."
                    )
                else:
                    warnings.append(
                        f"Multi-GPU detectado ({n_gpus} GPUs). El modelo usara "
                        f"device_map='auto' para repartirse entre las GPUs."
                    )

        if num_examples < 200:
            warnings.append(
                "Dataset muy pequeño (<200 ejemplos). El adapter puede overfittear. "
                "Recomendado: usar data augmentation o recopilar más datos."
            )

        if num_classes > 1 and num_examples / max(num_classes, 1) < 15:
            warnings.append(
                f"Pocos ejemplos por clase (~{num_examples // max(num_classes, 1)}). "
                "Algunas clases pueden no aprenderse correctamente."
            )

        if epochs >= 5:
            warnings.append(
                f"Épocas altas ({epochs}) con dataset pequeño → riesgo de overfitting. "
                "Monitoriza eval_loss y usa early stopping si es posible."
            )

        return warnings

    # ------------------------------------------------------------------
    # Detección de adapters reutilizables (S3.2)
    # ------------------------------------------------------------------

    @staticmethod
    def find_related_adapters(
        model_id: str,
        adapters_dir: str = "adapters",
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Escanea adapters/ y busca adapters relacionados con el modelo o dominio.

        Parámetros
        ----------
        model_id : str
            ID del modelo base actual (ej: "Qwen/Qwen2.5-14B-Instruct").
        adapters_dir : str
            Carpeta donde se almacenan los adapters.
        domain : str, opcional
            Dominio del dataset actual (ej: "medical", "legal").
            Si es None, solo compara por modelo base.

        Devuelve
        --------
        dict con:
            action: "incremental" | "transfer" | "from_scratch"
            suggestion: str — recomendación legible
            related: list[dict] — adapters relacionados con metadatos
        """
        adapters_path = Path(adapters_dir)
        if not adapters_path.exists():
            return {
                "action": "from_scratch",
                "suggestion": "No hay carpeta de adapters. Primer entrenamiento.",
                "related": [],
            }

        import json as _json
        same_base: List[Dict[str, Any]] = []
        same_domain: List[Dict[str, Any]] = []
        all_adapters: List[Dict[str, Any]] = []

        for adapter_path in sorted(adapters_path.iterdir()):
            if not adapter_path.is_dir():
                continue
            meta_file = adapter_path / "meta.json"
            if not meta_file.exists():
                continue

            try:
                meta = _json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            info = {
                "name": adapter_path.name,
                "path": str(adapter_path),
                "model_id": meta.get("model_id", "unknown"),
                "lora_r": meta.get("lora_r"),
                "train_loss": meta.get("train_loss"),
                "eval_loss": meta.get("eval_loss"),
                "elapsed_min": meta.get("elapsed_min"),
                "train_samples": meta.get("train_samples"),
            }
            all_adapters.append(info)

            # ¿Mismo modelo base?
            if meta.get("model_id") == model_id:
                same_base.append(info)

            # ¿Mismo dominio? (se infiere del nombre de carpeta o campo domain)
            adapter_domain = meta.get("domain") or meta.get("detected_domain")
            if not adapter_domain:
                # Inferir del nombre: "finance_sentiment_14b" → "finance"
                name_lower = adapter_path.name.lower()
                for d in ["medical", "legal", "financial", "technical", "general"]:
                    if d in name_lower:
                        adapter_domain = d
                        break
            if domain and adapter_domain == domain:
                same_domain.append(info)

        # ── Decidir acción ───────────────────────────────────────────
        if same_base:
            best = same_base[0]
            action = "incremental"
            suggestion = (
                f"Ya existe '{best['name']}' sobre el MISMO modelo base "
                f"({model_id}). Puedes hacer fine-tuning incremental "
                f"con r={best['lora_r']} y el mismo base model."
            )
            if best.get("train_loss"):
                suggestion += f" (train_loss={best['train_loss']:.3f})"
        elif same_domain:
            best = same_domain[0]
            action = "transfer"
            suggestion = (
                f"Adapter de dominio '{domain}' encontrado ('{best['name']}') "
                f"pero con modelo base distinto ({best['model_id']}). "
                "No se puede reutilizar directamente. Entrena desde cero."
            )
        elif all_adapters:
            action = "from_scratch"
            other_names = ", ".join(a["name"] for a in all_adapters[:3])
            suggestion = (
                f"Hay {len(all_adapters)} adapters ({other_names}) "
                f"pero ninguno comparte modelo base ni dominio. "
                "Entrena desde cero."
            )
        else:
            action = "from_scratch"
            suggestion = "No hay adapters existentes. Primer entrenamiento."

        return {
            "action": action,
            "suggestion": suggestion,
            "related": same_base or same_domain,
            "all_adapters": all_adapters,
        }

    def _classify_chat_template(self, raw_template: str) -> Optional[str]:
        """
        Clasifica el chat template a partir de su contenido raw (Jinja2).
        Detecta las firmas características de cada formato.
        """
        t = raw_template.lower()
        if "<|im_start|>" in t:
            return "chatml"
        if "<|begin_of_text|>" in t or "llama3" in t:
            return "llama3"
        if "<start_of_turn>" in t:
            return "gemma"
        if "<|user|>" in t and "phi" in t:
            return "phi3"
        if "[inst]" in t or "[/inst]" in t:
            return "alpaca"
        return None

    def _build_lora_config(
        self,
        family_info: dict,
        is_vlm: bool,
    ) -> Dict[str, Any]:
        """
        Construye el dict de configuración LoRA recomendado para este modelo.
        Compatible directamente con peft.LoraConfig(**lora_config).
        Usa strings para task_type si peft no está instalado.
        """
        # Usar strings directamente — no importar peft aquí.
        # peft dispara un scan lento de importlib.metadata al importarse,
        # lo que bloquea el arranque innecesariamente en Windows.
        task_type = (
            "FEATURE_EXTRACTION" if family_info["engine"] == "peft_trl"
            else "CAUSAL_LM"
        )

        return {
            "r":               self.lora_r,
            "lora_alpha":      self.lora_alpha,
            "target_modules":  family_info["target_modules"],
            "lora_dropout":    0.05,
            "bias":            "none",
            "task_type":       task_type,
        }
