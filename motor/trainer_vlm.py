"""
motor.trainer_vlm
=================
VLMTrainer: entrena adapters LoRA sobre Vision-Language Models (VLMs).

Modelos soportados (probados)
------------------------------
  Qwen2-VL  → Qwen/Qwen2-VL-2B-Instruct  (recomendado para GPUs ~11 GB)
              Qwen/Qwen2-VL-7B-Instruct  (~11 GB en 4-bit, bordeando GTX 1080 Ti)

Formato de dataset (ChatML multimodal)
--------------------------------------
Usa el campo 'messages' igual que LLMTrainer, pero el contenido del usuario
puede ser una lista mixta de imagen + texto:

  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "/ruta/absoluta/imagen.jpg"},
          {"type": "text",  "text": "¿Qué ves en la imagen?"}
        ]
      },
      {
        "role": "assistant",
        "content": "Descripción de la imagen."
      }
    ]
  }

Si no hay imágenes, content puede ser directamente un string (retrocompatible
con el formato de LLMTrainer para texto puro).

Generado por DataDigestor.from_images_folder_vlm() o manualmente.

Diferencias clave respecto a LLMTrainer
----------------------------------------
  - Usa AutoModelForVision2Seq en lugar de AutoModelForCausalLM.
  - Usa AutoProcessor (imagen + texto) en lugar de AutoTokenizer.
  - No usa TRL SFTTrainer (que no maneja imágenes): usa Trainer estándar
    con un DataCollator multimodal personalizado.
  - El LoRA se inyecta SOLO en el LLM backbone; el vision encoder queda
    congelado para ahorrar VRAM y tiempo.
  - batch_size=1 por defecto (imágenes consumen mucha VRAM adicional).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Helpers de entorno
# ---------------------------------------------------------------------------

def _get_compute_capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


def _supports_bf16() -> bool:
    return _get_compute_capability()[0] >= 8


def _args_seleccion_checkpoint(
    keep_best: bool,
    eval_ds,
    eval_every_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Política de evaluación y guardado: cuándo se mide y con qué punto se queda.

    El Trainer ya evaluaba cada época (`eval_strategy="epoch"`), pero con
    `save_strategy="no"` y sin `load_best_model_at_end` tiraba esa medición y
    guardaba siempre la final — aunque su propia evaluación dijera que era la
    peor. Medido en EXIST-VLM: se guardó un adapter de eval_loss 0.2652
    habiendo pasado por 0.1658 en la primera época.

    `eval_every_steps` sube la RESOLUCIÓN de esa elección. Evaluando por época
    solo se puede elegir el punto 1.0, 2.0, 3.0…; si el óptimo cae en medio
    (en EXIST-VLM la eval_loss subía ya entre la época 1 y la 2), se pierde.
    Evaluar cada N pasos permite quedarse con el mínimo real de la MISMA
    trayectoria, sin alargar el entrenamiento ni tocar el planificador de LR.

    Elegir la mejor exige una métrica que comparar: sin dataset de evaluación
    no se puede, y se avisa en vez de fingir que sí.
    """
    por_pasos = eval_every_steps is not None and eval_every_steps > 0
    ritmo = (
        {"eval_strategy": "steps", "eval_steps": eval_every_steps}
        if por_pasos
        else {"eval_strategy": "epoch"}
    )

    puede_elegir = bool(keep_best) and eval_ds is not None and len(eval_ds) > 0

    if not puede_elegir:
        if keep_best:
            print("[VLMTrainer] AVISO: sin dataset de evaluación no se puede elegir "
                  "el mejor punto; se guardará el último.")
        return {**ritmo, "save_strategy": "no"}

    # save_strategy DEBE casar con eval_strategy o transformers rechaza
    # load_best_model_at_end.
    guardado = (
        {"save_strategy": "steps", "save_steps": eval_every_steps}
        if por_pasos
        else {"save_strategy": "epoch"}
    )

    return {
        **ritmo,
        **guardado,
        "load_best_model_at_end": True,
        "metric_for_best_model":  "eval_loss",
        "greater_is_better":      False,
        "save_total_limit":       1,
    }


# ---------------------------------------------------------------------------
# Data Collator multimodal
# ---------------------------------------------------------------------------

class _VLMDataCollator:
    """
    Collator para VLMs multimodales.

    Convierte cada ejemplo {"messages": [...]} en tensores listos para Trainer.
    Maneja imágenes como PIL.Image cargadas desde rutas en disco.

    Labels: copia de input_ids con -100 en posiciones de padding. Con
    mask_prompt=True, además se enmascara el prompt entero (completion-only
    loss): el gradiente viene solo de la respuesta del assistant.
    """

    def __init__(self, processor, max_seq_length: int = 1024, mask_prompt: bool = False):
        self.processor      = processor
        self.max_seq_length = max_seq_length
        self.mask_prompt    = mask_prompt
        self._mask_warned   = False

    # --- extrae imágenes PIL de los mensajes ---
    def _extract_images(self, messages: list) -> list:
        from PIL import Image as PILImage

        images = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img_path = part.get("image", "")
                        if img_path:
                            p = Path(img_path)
                            if p.exists():
                                images.append(PILImage.open(p).convert("RGB"))
                            else:
                                # Imagen no encontrada: placeholder blanco
                                images.append(PILImage.new("RGB", (224, 224), (255, 255, 255)))
        return images

    # --- aplica chat template o fallback ---
    def _apply_chat_template(self, messages: list) -> str:
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            tok = getattr(self.processor, "tokenizer", self.processor)
            parts = []
            for msg in messages:
                role    = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                parts.append(f"<|{role}|>\n{content}")
            return "\n".join(parts) + tok.eos_token

    # --- nº de tokens que ocupa el prompt (todo menos la respuesta final) ---
    def _prompt_token_len(self, messages: list, image) -> Optional[int]:
        """
        Longitud en tokens del prompt: los mensajes SIN el turno final del
        assistant, con add_generation_prompt=True.

        Se procesa CON LA MISMA IMAGEN a propósito: los VLM tipo Qwen2-VL
        expanden el placeholder de imagen a N tokens según su resolución, así
        que contar sin ella daría un corte desplazado y enmascararía parte de
        la respuesta (o dejaría prompt sin enmascarar).

        Devuelve None si no se puede determinar (se omite el enmascarado).
        """
        prompt_msgs = list(messages)
        if prompt_msgs and prompt_msgs[-1].get("role") == "assistant":
            prompt_msgs = prompt_msgs[:-1]
        if not prompt_msgs:
            return None

        try:
            text = self.processor.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            if image is not None:
                enc = self.processor(text=[text], images=[image], return_tensors="pt")
            else:
                tok = getattr(self.processor, "tokenizer", self.processor)
                enc = tok([text], return_tensors="pt")
            return int(enc["input_ids"].shape[1])
        except Exception:
            return None

    def __call__(self, examples: list) -> dict:
        from PIL import Image as PILImage

        texts_out   = []
        images_out  = []

        for ex in examples:
            messages = ex["messages"]
            texts_out.append(self._apply_chat_template(messages))
            images_out.append(self._extract_images(messages))

        has_images = any(len(imgs) > 0 for imgs in images_out)
        tok = getattr(self.processor, "tokenizer", self.processor)

        if has_images:
            # Una imagen por ejemplo (primera). Placeholder si no hay.
            flat_images = [
                imgs[0] if imgs else PILImage.new("RGB", (224, 224))
                for imgs in images_out
            ]
            batch = self.processor(
                text           = texts_out,
                images         = flat_images,
                padding        = True,
                truncation     = True,
                max_length     = self.max_seq_length,
                return_tensors = "pt",
            )
        else:
            batch = tok(
                texts_out,
                padding        = True,
                truncation     = True,
                max_length     = self.max_seq_length,
                return_tensors = "pt",
            )

        # Labels: -100 en padding
        labels = batch["input_ids"].clone()
        pad_id = tok.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Completion-only loss (opcional): además, -100 en TODO el prompt, de
        # modo que el gradiente venga solo de la respuesta del assistant.
        # Sin esto, la pérdida cubre prompt + tokens de imagen + respuesta; en
        # tareas con respuesta corta (p.ej. una etiqueta YES/NO frente a un
        # prompt de cientos de tokens) la señal útil queda diluida.
        if self.mask_prompt:
            imgs_for_len = flat_images if has_images else [None] * len(examples)
            pad_left     = getattr(tok, "padding_side", "right") == "left"
            total_len    = labels.shape[1]

            for i, ex in enumerate(examples):
                n_prompt = self._prompt_token_len(ex["messages"], imgs_for_len[i])
                if n_prompt is None:
                    continue

                if pad_left and pad_id is not None:
                    # Padding a la izquierda (lo que usa Qwen2-VL): el relleno es
                    # un PREFIJO. Se cuentan los pads de cabecera, no los pads
                    # totales: si el modelo reutiliza el token de pad como eos,
                    # contar todos los pads metería el eos final en la cuenta y
                    # desplazaría el corte sobre la respuesta.
                    no_pad = (batch["input_ids"][i] != pad_id).nonzero().flatten()
                    n_lead_pad = int(no_pad[0]) if len(no_pad) else total_len
                    cut = n_lead_pad + n_prompt
                else:
                    cut = n_prompt
                cut = min(cut, total_len)

                # Si la truncación se comió la respuesta, enmascarar dejaría el
                # ejemplo entero a -100 → loss NaN. Mejor dejarlo sin enmascarar.
                if cut >= total_len or bool((labels[i, cut:] == -100).all()):
                    if not self._mask_warned:
                        print("[VLMTrainer] AVISO: algún ejemplo se queda sin tokens "
                              "de respuesta al enmascarar el prompt (¿max_seq_length "
                              "demasiado corto?). Esos ejemplos se entrenan SIN "
                              "enmascarar para no romper la pérdida.")
                        self._mask_warned = True
                    continue

                labels[i, :cut] = -100

        batch["labels"] = labels

        return batch


# ---------------------------------------------------------------------------
# VLMTrainer
# ---------------------------------------------------------------------------

class VLMTrainer:
    """
    Entrena un adapter LoRA sobre un Vision-Language Model (VLM).

    Usa PEFT + transformers Trainer (no TRL SFTTrainer, que no maneja imágenes).
    El LoRA se aplica SOLO al LLM backbone; el vision encoder queda congelado.

    Parámetros
    ----------
    model_id : str
        ID HuggingFace del VLM. Ej: "Qwen/Qwen2-VL-2B-Instruct"
    target_modules : list[str], opcional
        Capas donde inyectar LoRA. Por defecto: attention + MLP del LLM.
    lora_r : int
        Rango LoRA. Por defecto 16.
    lora_alpha : int
        Alpha LoRA. Por defecto 32.
    lora_dropout : float
        Dropout LoRA. Por defecto 0.05.
    load_in_4bit : bool
        Cuantización 4-bit NF4. Por defecto True.
    max_seq_length : int
        Máx. tokens. Las imágenes consumen muchos tokens; default 1024.
    cache_dir : str, opcional
        Caché HuggingFace para los pesos.
    mask_prompt : bool
        Completion-only loss: si True, el prompt se enmascara (-100) y el
        gradiente viene SOLO de la respuesta del assistant. Por defecto False
        (comportamiento histórico: la pérdida cubre toda la secuencia).
        Actívalo cuando la respuesta sea corta frente al prompt —p. ej. una
        etiqueta de clasificación—, donde si no la señal queda diluida.
    """

    # Módulos de atención + MLP del backbone LLM (comunes a Qwen2-VL, LLaVA, InternVL…)
    DEFAULT_TARGET_MODULES = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def __init__(
        self,
        model_id: str,
        target_modules: Optional[List[str]] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        max_seq_length: int = 1024,
        cache_dir: Optional[str] = None,
        mask_prompt: bool = False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[VLMTrainer] No se detectó GPU CUDA.\n"
                "El entrenamiento LoRA requiere una GPU NVIDIA con CUDA."
            )

        self.model_id       = model_id
        self.target_modules = target_modules or self.DEFAULT_TARGET_MODULES
        self.lora_r         = lora_r
        self.lora_alpha     = lora_alpha
        self.lora_dropout   = lora_dropout
        self.load_in_4bit   = load_in_4bit
        self.max_seq_length = max_seq_length
        self.cache_dir      = cache_dir
        self.mask_prompt    = mask_prompt

        self._use_bf16 = _supports_bf16()

        major, minor = _get_compute_capability()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

        print("=" * 60)
        print("  VLMTrainer — configuración detectada")
        print("=" * 60)
        print(f"  GPU:            {gpu_name}")
        print(f"  VRAM:           {vram_gb:.1f} GB")
        print(f"  Compute cap.:   {major}.{minor}")
        print(f"  bf16:           {'Sí (Ampere+)' if self._use_bf16 else 'No — fp16 (Pascal)'}")
        print(f"  Modo:           PEFT + Trainer (VLM multimodal)")
        print(f"  Target modules: {self.target_modules}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset_path: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 1,
        grad_accum: int = 16,
        learning_rate: float = 2e-4,
        warmup_steps: int = 10,
        eval_split: float = 0.1,
        logging_steps: int = 10,
        keep_best: bool = True,
        eval_every_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Entrena el adapter LoRA VLM y lo guarda en output_dir.

        Parámetros
        ----------
        dataset_path : str
            Ruta al .jsonl generado por DataDigestor.from_images_folder_vlm().
        output_dir : str
            Carpeta donde se guardará el adapter.
        epochs : int
            Épocas. Por defecto 3.
        batch_size : int
            Batch por GPU. Por defecto 1 (imágenes consumen mucha VRAM).
            Batch efectivo = batch_size × grad_accum (default: 1 × 16 = 16).
        grad_accum : int
            Acumulación de gradiente. Por defecto 16.
        learning_rate : float
            Learning rate. Por defecto 2e-4.
        warmup_steps : int
            Warmup lineal. Por defecto 10.
        eval_split : float
            Fracción para validación. Por defecto 0.1.
        logging_steps : int
            Frecuencia de log. Por defecto 10.
        keep_best : bool
            Guardar la MEJOR época según eval_loss en vez de la última. Por
            defecto True. El Trainer ya evaluaba cada época; sin esto tiraba
            la medición y guardaba la final aunque fuese la peor. Necesita
            dataset de evaluación (si no lo hay, avisa y guarda la última).
        eval_every_steps : int, opcional
            Evaluar (y poder guardar) cada N pasos en vez de cada época. Sube la
            RESOLUCIÓN de `keep_best`: por época solo se puede elegir el punto
            1.0, 2.0, 3.0…; si el óptimo cae en medio, se pierde. No alarga el
            entrenamiento ni toca el planificador de LR — solo mide más a menudo
            (cada evaluación cuesta una pasada por el split de eval).

        Devuelve
        --------
        dict con métricas: train_loss, eval_loss, tiempo, VRAM peak, etc.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        meta_path = output_dir / "meta.json"
        if meta_path.exists():
            print(f"[VLMTrainer] Checkpoint encontrado en {output_dir} — saltando entrenamiento.")
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)

        print(f"\n[VLMTrainer] Cargando modelo VLM: {self.model_id}")
        model, processor = self._load_model()

        print(f"\n[VLMTrainer] Cargando dataset: {dataset_path}")
        train_ds, eval_ds = self._load_dataset(dataset_path, eval_split)
        print(f"  Train: {len(train_ds)} ejemplos  |  Eval: {len(eval_ds)} ejemplos")

        collator = _VLMDataCollator(
            processor      = processor,
            max_seq_length = self.max_seq_length,
            mask_prompt    = self.mask_prompt,
        )

        print(f"\n[VLMTrainer] Iniciando entrenamiento VLM...")
        print(f"  Épocas: {epochs}  |  Batch efectivo: {batch_size * grad_accum}")
        print(f"  LR: {learning_rate}  |  max_seq_length: {self.max_seq_length}")
        print(f"  Pérdida: {'solo respuesta (prompt enmascarado)' if self.mask_prompt else 'secuencia completa'}")
        _ritmo = f"cada {eval_every_steps} pasos" if eval_every_steps else "cada época"
        print(f"  Evaluación: {_ritmo}")
        print(f"  Se guarda: {'el MEJOR punto (por eval_loss)' if keep_best else 'el último'}")

        t0 = time.time()
        trainer = self._build_trainer(
            model         = model,
            train_ds      = train_ds,
            eval_ds       = eval_ds,
            collator      = collator,
            output_dir    = str(output_dir / "checkpoints"),
            epochs        = epochs,
            batch_size    = batch_size,
            grad_accum    = grad_accum,
            learning_rate = learning_rate,
            warmup_steps  = warmup_steps,
            logging_steps = logging_steps,
            keep_best     = keep_best,
            eval_every_steps = eval_every_steps,
        )

        train_result = trainer.train()
        elapsed = time.time() - t0

        # Con keep_best, el modelo en memoria ya es el de la mejor época: el
        # evaluate() de abajo mide ESE, no el de la última.
        mejor = getattr(trainer.state, "best_metric", None)
        if mejor is not None:
            print(f"[VLMTrainer] Mejor época por eval_loss: {round(mejor, 4)} "
                  f"(es la que se guarda)")

        print("\n[VLMTrainer] Evaluando...")
        eval_result = trainer.evaluate()

        print(f"\n[VLMTrainer] Guardando adapter en {output_dir}...")
        trainer.model.save_pretrained(str(output_dir))
        processor.save_pretrained(str(output_dir / "processor"))

        metrics = {
            "model_id":        self.model_id,
            "model_type":      "vlm",
            "adapter_dir":     str(output_dir),
            "engine":          "peft_trainer",
            "lora_r":          self.lora_r,
            "lora_alpha":      self.lora_alpha,
            "target_modules":  self.target_modules,
            "epochs":          epochs,
            "batch_effective": batch_size * grad_accum,
            "train_loss":      round(train_result.training_loss, 4),
            "eval_loss":       round(eval_result.get("eval_loss", 0.0), 4),
            "keep_best":       bool(keep_best),
            "best_eval_loss":  round(mejor, 4) if mejor is not None else None,
            "best_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
            "eval_every_steps": eval_every_steps,
            "mask_prompt":     bool(self.mask_prompt),
            "train_samples":   len(train_ds),
            "eval_samples":    len(eval_ds),
            "elapsed_min":     round(elapsed / 60, 2),
            "vram_peak_gb":    round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("  Entrenamiento VLM completado")
        print("=" * 60)
        print(f"  Train loss:   {metrics['train_loss']}")
        print(f"  Eval loss:    {metrics['eval_loss']}")
        print(f"  Tiempo:       {metrics['elapsed_min']} min")
        print(f"  VRAM pico:    {metrics['vram_peak_gb']} GB")
        print(f"  Adapter:      {output_dir}")
        print("=" * 60)

        return metrics

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------

    def _load_model(self):
        """Carga el VLM con 4-bit NF4 y aplica LoRA al backbone LLM."""
        from transformers import AutoProcessor, BitsAndBytesConfig
        # transformers >= 5 renombró AutoModelForVision2Seq → AutoModelForImageTextToText
        try:
            from transformers import AutoModelForVision2Seq as _AutoVLMModel
        except ImportError:
            from transformers import AutoModelForImageTextToText as _AutoVLMModel
        from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

        print("  Cargando procesador (tokenizer + image processor)...")
        processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir         = self.cache_dir,
            trust_remote_code = True,
        )

        # Asegurar pad_token
        tok = getattr(processor, "tokenizer", processor)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        _dtype = torch.bfloat16 if self._use_bf16 else torch.float16
        model_kwargs = {
            "device_map":        "auto",
            "cache_dir":         self.cache_dir,
            "trust_remote_code": True,
        }

        if self.load_in_4bit:
            from motor._model_utils import apply_4bit_quantization
            if not apply_4bit_quantization(model_kwargs, dtype=_dtype):
                model_kwargs["torch_dtype"] = _dtype
        else:
            model_kwargs["torch_dtype"] = _dtype

        print(f"  Cargando pesos: {self.model_id}")
        model = _AutoVLMModel.from_pretrained(self.model_id, **model_kwargs)

        if self.load_in_4bit and "quantization_config" in model_kwargs:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        else:
            model.gradient_checkpointing_enable()

        # Fix Pascal (compute < 8.0): bf16 → fp16
        if not self._use_bf16:
            n_cast = 0
            for param in model.parameters():
                if param.dtype == torch.bfloat16:
                    param.data = param.data.to(torch.float16)
                    n_cast += 1
            if n_cast:
                print(f"  [INFO] {n_cast} tensores bf16 → fp16 (GPU Pascal, compute < 8.0)")

        # LoRA solo en el LLM backbone (el vision encoder queda congelado)
        lora_config = LoraConfig(
            r              = self.lora_r,
            lora_alpha     = self.lora_alpha,
            target_modules = self.target_modules,
            lora_dropout   = self.lora_dropout,
            bias           = "none",
            task_type      = TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        try:
            model.print_trainable_parameters()
        except AttributeError:
            total     = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parámetros entrenables: {trainable:,} / {total:,} ({trainable/total*100:.3f}%)")

        return model, processor

    # ------------------------------------------------------------------
    # Carga del dataset
    # ------------------------------------------------------------------

    def _load_dataset(self, dataset_path: str, eval_split: float):
        from datasets import load_dataset

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"[VLMTrainer] Dataset no encontrado: {dataset_path}\n"
                "Genera el dataset con DataDigestor.from_images_folder_vlm()"
            )

        ds = load_dataset("json", data_files=str(path), split="train")

        if "messages" not in ds.column_names:
            raise ValueError(
                "[VLMTrainer] El dataset necesita el campo 'messages' en formato ChatML.\n"
                "Usa DataDigestor.from_images_folder_vlm() para generarlo."
            )

        split = ds.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]

    # ------------------------------------------------------------------
    # Construcción del Trainer
    # ------------------------------------------------------------------

    def _build_trainer(
        self,
        model,
        train_ds,
        eval_ds,
        collator,
        output_dir: str,
        epochs: int,
        batch_size: int,
        grad_accum: int,
        learning_rate: float,
        warmup_steps: int,
        logging_steps: int,
        keep_best: bool = True,
        eval_every_steps: Optional[int] = None,
    ):
        from transformers import Trainer, TrainingArguments, TrainerCallback

        class _ProgressCB(TrainerCallback):
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs
                self.start_t      = time.time()
                self.total_steps  = 0

            def on_train_begin(self, args, state, control, **kw):
                self.total_steps = state.max_steps
                print(f"\n[Progreso] Épocas: {self.total_epochs} | Pasos totales: {self.total_steps}")
                print("-" * 65)
                sys.stdout.flush()

            def on_log(self, args, state, control, logs=None, **kw):
                if not logs:
                    return
                step      = state.global_step
                loss      = logs.get("loss") or logs.get("train_loss")
                eval_loss = logs.get("eval_loss")
                elapsed   = time.time() - self.start_t

                if self.total_steps > 0 and step > 0:
                    eta     = elapsed / step * (self.total_steps - step)
                    eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
                else:
                    eta_str = "?"

                vram_str = (
                    f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"
                    if torch.cuda.is_available() else "N/A"
                )
                pct = int(step / self.total_steps * 100) if self.total_steps else 0

                if eval_loss is not None:
                    print(
                        f"  === Eval | paso {step} "
                        f"| eval_loss={eval_loss:.4f} "
                        f"| elapsed={int(elapsed // 60)}m ==="
                    )
                elif loss is not None:
                    print(
                        f"  [Paso {step}/{self.total_steps} | {pct}%] "
                        f"loss={loss:.4f} | VRAM={vram_str} | ETA {eta_str}"
                    )
                sys.stdout.flush()

            def on_train_end(self, args, state, control, **kw):
                elapsed = time.time() - self.start_t
                print("-" * 65)
                print(f"[Progreso] Completado en {int(elapsed // 60)}m {int(elapsed % 60)}s")
                sys.stdout.flush()

        seleccion = _args_seleccion_checkpoint(keep_best, eval_ds, eval_every_steps)

        training_args = TrainingArguments(
            output_dir                  = output_dir,
            **seleccion,
            num_train_epochs            = epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            gradient_accumulation_steps = grad_accum,
            gradient_checkpointing      = True,
            warmup_steps                = warmup_steps,
            learning_rate               = learning_rate,
            # Fix Pascal: no fp16/bf16 en los args del Trainer.
            # bitsandbytes gestiona fp16 internamente via bnb_4bit_compute_dtype.
            fp16                        = False,
            bf16                        = self._use_bf16,
            logging_steps               = logging_steps,
            report_to                   = "none",
            seed                        = 42,
            disable_tqdm                = True,
            # CRÍTICO para VLMs: preservar pixel_values y otros campos de imagen
            remove_unused_columns       = False,
            # Evita OOM con imágenes grandes en memoria
            dataloader_pin_memory       = False,
        )

        return Trainer(
            model         = model,
            args          = training_args,
            train_dataset = train_ds,
            eval_dataset  = eval_ds,
            data_collator = collator,
            callbacks     = [_ProgressCB(total_epochs=epochs)],
        )
