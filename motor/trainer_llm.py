"""
motor.trainer_llm
=================
LLMTrainer: entrena adapters LoRA sobre LLMs de texto (causal LM).

Selección automática de rama según GPU disponible:

  Rama A          → Unsloth (GPU Ampere+, compute ≥ 8.0)
                    2× más rápido, 70% menos VRAM, kernels Triton optimizados.
                    Compatible con: RTX 3090, 4080, 4090, A100, H100.

  Rama A-fallback → PEFT + TRL SFTTrainer (cualquier GPU CUDA, Pascal+)
                    Compatible con: GTX 1080 Ti, RTX 2080, cualquier GPU antigua.
                    Mismo adapter de salida, ~1.5× más lento.

El trainer detecta automáticamente cuál usar. Si Unsloth no está instalado
o la GPU no es compatible, cae al fallback sin intervención del usuario.

Uso básico
----------
>>> from motor.trainer_llm import LLMTrainer
>>> trainer = LLMTrainer(
...     model_id       = "Qwen/Qwen2.5-3B-Instruct",
...     target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
...                        "gate_proj", "up_proj", "down_proj"],
... )
>>> metrics = trainer.fit(
...     dataset_path = "data/titanic_dataset.jsonl",
...     output_dir   = "adapters/titanic_llm/",
...     epochs       = 3,
... )
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# ===========================================================================
# Helpers de detección de entorno
# ===========================================================================

# Arquitecturas multimodales "unified" (encoder-free) que generan texto pero
# NO se cargan con AutoModelForCausalLM/AutoTokenizer sino con
# AutoModelForMultimodalLM/AutoProcessor. Detectadas por model_type del
# config.json crudo (sin depender de que transformers conozca la arquitectura).
_MULTIMODAL_LM_TYPES = {"gemma4_unified", "gemma4_unified_text"}


def _get_compute_capability() -> tuple[int, int]:
    """Devuelve (major, minor) de la GPU actual."""
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


def _supports_bf16() -> bool:
    """bf16 solo está soportado en Ampere+ (compute ≥ 8.0)."""
    major, _ = _get_compute_capability()
    return major >= 8


def _check_unsloth_available() -> bool:
    """
    Devuelve True si Unsloth está instalado Y la GPU es compatible (Ampere+).
    """
    try:
        import unsloth  # noqa: F401
        major, _ = _get_compute_capability()
        return major >= 8
    except ImportError:
        return False


def _format_dataset(dataset, tokenizer):
    """
    Aplica el chat template del tokenizador a cada ejemplo del dataset.
    Convierte {"messages": [...]} → {"text": "<cadena formateada>"}.
    Necesario para SFTTrainer con dataset_text_field="text".
    """
    def _apply(examples):
        texts = []
        for msgs in examples["messages"]:
            try:
                text = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Fallback manual si el tokenizador no tiene chat template
                parts = []
                for m in msgs:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if role == "system":
                        parts.append(f"<|system|>\n{content}")
                    elif role == "user":
                        parts.append(f"<|user|>\n{content}")
                    elif role == "assistant":
                        parts.append(f"<|assistant|>\n{content}")
                text = "\n".join(parts) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    return dataset.map(_apply, batched=True, remove_columns=dataset.column_names)


# ===========================================================================
# Callback de progreso en tiempo real (funciona bien sobre SSH)
# ===========================================================================

class _LiveProgressCallback:
    """
    Callback para HuggingFace Trainer que imprime progreso en tiempo real.
    Diseñado para SSH: sin barras de progreso ANSI, solo texto plano.

    Muestra por cada logging_step:
      [Época X/Y | Paso ZZZ/TTT | Z%] loss=X.XXXX | lr=X.Xe-05 | VRAM=X.XGB | ETA Xm Xs
    Y al final de cada época:
      === Época X completada | train_loss=X.XXXX | eval_loss=X.XXXX ===
    """

    def __init__(self, total_epochs: int):
        self.total_epochs   = total_epochs
        self.epoch          = 0
        self.step           = 0
        self.total_steps    = 0
        self.epoch_start_t  = time.time()
        self.train_start_t  = time.time()
        self.last_loss      = None
        self.last_lr        = None

    # --- Interfaz compatible con transformers.TrainerCallback ---

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps   = state.max_steps
        self.train_start_t = time.time()
        print(f"\n[Progreso] Iniciando entrenamiento:")
        print(f"  Épocas:       {self.total_epochs}")
        print(f"  Pasos totales:{self.total_steps}")
        print(f"  Batch ef.:    {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"  Precisión:    {'bf16' if args.bf16 else 'fp16'}")
        print("-" * 65)
        sys.stdout.flush()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch        = int(state.epoch) + 1
        self.epoch_start_t = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        self.step     = state.global_step
        loss          = logs.get("loss") or logs.get("train_loss")
        lr            = logs.get("learning_rate")
        eval_loss     = logs.get("eval_loss")

        if loss is not None:
            self.last_loss = loss
        if lr is not None:
            self.last_lr = lr

        # Solo imprimir en pasos de entrenamiento (no en eval)
        if loss is None and eval_loss is None:
            return

        # Calcular ETA
        elapsed   = time.time() - self.train_start_t
        if self.total_steps > 0 and self.step > 0:
            secs_per_step = elapsed / self.step
            remaining     = secs_per_step * (self.total_steps - self.step)
            eta_str = f"{int(remaining // 60)}m {int(remaining % 60)}s"
        else:
            eta_str = "?"

        # VRAM actual
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_str  = f"{vram_used:.1f}GB"
        else:
            vram_str = "N/A"

        pct = int(self.step / self.total_steps * 100) if self.total_steps > 0 else 0

        if eval_loss is not None:
            # Log de evaluación al final de época
            print(
                f"  === Época {self.epoch}/{self.total_epochs} completada "
                f"| eval_loss={eval_loss:.4f} "
                f"| elapsed={int(elapsed // 60)}m {int(elapsed % 60)}s ==="
            )
        else:
            # Log de paso de entrenamiento
            loss_str = f"{loss:.4f}" if loss is not None else "?"
            lr_str   = f"{lr:.2e}" if lr is not None else "?"
            print(
                f"  [Época {self.epoch}/{self.total_epochs} "
                f"| Paso {self.step}/{self.total_steps} | {pct}%] "
                f"loss={loss_str} | lr={lr_str} | VRAM={vram_str} | ETA {eta_str}"
            )
        sys.stdout.flush()

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start_t
        print("-" * 65)
        print(f"[Progreso] Entrenamiento completado en {int(elapsed // 60)}m {int(elapsed % 60)}s")
        if self.last_loss is not None:
            print(f"  Último train_loss: {self.last_loss:.4f}")
        sys.stdout.flush()


# ===========================================================================
# LLMTrainer
# ===========================================================================

class LLMTrainer:
    """
    Entrena un adapter LoRA sobre un LLM de texto (causal LM).

    Parámetros
    ----------
    model_id : str
        ID del modelo HuggingFace o ruta local.
        Ej: "Qwen/Qwen2.5-3B-Instruct"
    target_modules : list[str]
        Capas donde inyectar LoRA. El ModelAnalyzer las determina automáticamente.
    lora_r : int
        Rango LoRA. Por defecto 16.
    lora_alpha : int
        Alpha LoRA. Por defecto 32.
    lora_dropout : float
        Dropout LoRA. Por defecto 0.05.
    load_in_4bit : bool
        Cuantización 4-bit (NF4). Reduce VRAM ~4×. Por defecto True.
    max_seq_length : int
        Longitud máxima de secuencia en tokens.
        2048 por defecto (RTX 4090 / GPUs >= 16 GB). Reducir a 512 si VRAM < 12 GB.
    cache_dir : str, opcional
        Directorio de caché para los pesos del modelo base.
    """

    def __init__(
        self,
        model_id: str,
        target_modules: Optional[List[str]] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        max_seq_length: int = 2048,
        cache_dir: Optional[str] = None,
    ):
        # ── Detectar hardware y ajustar configuración automáticamente ───────
        from motor.hardware import detect_hardware
        self._hw = detect_hardware()
        hw_kw   = self._hw.training_kwargs()

        # Si el usuario no pasó valores explícitos, usamos los del perfil
        # (los parámetros con valor=default se consideran "no explícitos")
        if max_seq_length == 2048:          # valor por defecto → auto-ajustar
            max_seq_length = hw_kw.get("max_seq_length", 2048)
        if load_in_4bit is True:            # valor por defecto → auto-ajustar
            load_in_4bit = hw_kw.get("load_in_4bit", True)

        self.model_id       = model_id
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.lora_r         = lora_r
        self.lora_alpha     = lora_alpha
        self.lora_dropout   = lora_dropout
        self.load_in_4bit   = load_in_4bit
        self.max_seq_length = max_seq_length
        self.cache_dir      = cache_dir

        # Modo CPU: forzar sin cuantización (bitsandbytes requiere CUDA)
        self._use_cpu       = hw_kw.get("use_cpu", False)
        self._cpu_offload   = hw_kw.get("cpu_offload", False)
        self._grad_ckpt     = hw_kw.get("gradient_checkpointing", False)

        if self._use_cpu or self._cpu_offload:
            self.load_in_4bit = False  # bitsandbytes no funciona sin CUDA

        # ¿Arquitectura multimodal unified (p.ej. gemma4_unified)? Entrena
        # como texto pero con carga distinta, y Unsloth no la soporta.
        self._multimodal_lm = self._detect_multimodal_lm()

        # Detección de rama (Unsloth solo en GPU Ampere+ y arquitecturas clásicas)
        self._use_unsloth = (
            _check_unsloth_available()
            and self._hw.has_usable_gpu
            and not self._multimodal_lm
        )
        self._use_bf16    = _supports_bf16() and not self._use_cpu

        # ── Informe de configuración ─────────────────────────────────────────
        print(self._hw)
        print(f"\n  LLMTrainer — perfil seleccionado: {self._hw.training_profile}")
        print(f"  max_seq_length:  {self.max_seq_length}")
        print(f"  load_in_4bit:    {self.load_in_4bit}")
        print(f"  use_bf16:        {self._use_bf16}")
        print(f"  gradient_ckpt:   {self._grad_ckpt}")
        if self._use_cpu:
            print(f"  Modo CPU:        SÍ  (RAM: {self._hw.ram_total_gb:.1f} GB)")
            if "recommended_model" in hw_kw and model_id.endswith("7B-Instruct"):
                print(f"  ⚠  CPU + 7B puede ser muy lento. Considera:")
                print(f"     {hw_kw['recommended_model']}")
        elif self._hw.has_usable_gpu:
            if self._use_unsloth:
                print(f"  Rama:            A — Unsloth (Ampere+)")
            else:
                maj = self._hw.compute_major
                print(f"  Rama:            A-fallback — PEFT+TRL"
                      f"  (compute {maj}.{self._hw.compute_minor})")
        else:
            print(f"  GPU:             no disponible / iGPU → modo CPU")
        print("=" * 60)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset_path: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 4,
        grad_accum: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 10,
        eval_split: float = 0.1,
        logging_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        Entrena el adapter LoRA y lo guarda en output_dir.

        Parámetros
        ----------
        dataset_path : str
            Ruta al archivo .jsonl generado por DataDigestor.
        output_dir : str
            Carpeta donde se guardará el adapter.
        epochs : int
            Número de épocas. Por defecto 3.
        batch_size : int
            Batch por GPU. Por defecto 4.
            Batch efectivo = batch_size × grad_accum (por defecto 16).
        grad_accum : int
            Pasos de acumulación de gradiente. Por defecto 4.
        learning_rate : float
            Learning rate. Por defecto 2e-4 (recomendado para LoRA).
        warmup_steps : int
            Pasos de warmup lineal. Por defecto 10.
        eval_split : float
            Fracción del dataset para validación. Por defecto 0.1 (10%).
        logging_steps : int
            Frecuencia de log de métricas. Por defecto 10.

        Devuelve
        --------
        dict
            Métricas del entrenamiento: train_loss, eval_loss, tiempo, etc.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-ajuste desde el perfil de hardware (si se pasan los valores por defecto)
        hw_kw = self._hw.training_kwargs()
        if batch_size == 4:
            batch_size = hw_kw.get("batch_size", batch_size)
        if grad_accum == 4:
            grad_accum = hw_kw.get("grad_accum", grad_accum)

        # Checkpoint: si ya existe el adapter, saltar
        meta_path = output_dir / "meta.json"
        if meta_path.exists():
            print(f"[LLMTrainer] Checkpoint encontrado en {output_dir} — saltando entrenamiento.")
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)

        print(f"\n[LLMTrainer] Cargando dataset: {dataset_path}")
        train_ds, eval_ds = self._load_dataset(dataset_path, eval_split)
        print(f"  Train: {len(train_ds)} ejemplos  |  Eval: {len(eval_ds)} ejemplos")

        print(f"\n[LLMTrainer] Cargando modelo base: {self.model_id}")
        # ── Resumen de hardware (ya mostrado en __init__, aquí solo VRAM) ────
        if self._hw.cuda_available and not self._use_cpu:
            try:
                import torch
                vram_free_gb = torch.cuda.mem_get_info()[0] / 1e9
                n_gpus       = torch.cuda.device_count()
                gpu_name     = self._hw.gpu_name
                print(f"  GPU: {gpu_name} | VRAM libre: {vram_free_gb:.1f} GB")
                if n_gpus > 1:
                    print(f"  Multi-GPU: {n_gpus} GPUs detectadas — usando device_map='auto'")
                elif vram_free_gb < 6.0:
                    print(f"  ⚠  VRAM libre baja ({vram_free_gb:.1f} GB). "
                          f"Si el modelo no cabe, considera:")
                    print(f"     • Usar un modelo más pequeño")
                    print(f"     • Reducir batch_size o aumentar grad_accum")
            except Exception:
                pass
        elif self._use_cpu:
            print(f"  Modo CPU — RAM disponible: {self._hw.ram_free_gb:.1f} GB")
        # ─────────────────────────────────────────────────────────────────────
        if self._use_unsloth:
            model, tokenizer = self._load_unsloth()
        else:
            model, tokenizer = self._load_peft_trl()

        print(f"\n[LLMTrainer] Aplicando chat template al dataset...")
        train_ds = _format_dataset(train_ds, tokenizer)
        eval_ds  = _format_dataset(eval_ds,  tokenizer)

        print(f"\n  Ejemplo de texto formateado (primeros 200 chars):")
        print(f"  {train_ds[0]['text'][:200]}...")

        print(f"\n[LLMTrainer] Iniciando entrenamiento...")
        print(f"  Épocas: {epochs}  |  Batch efectivo: {batch_size * grad_accum}")
        print(f"  LR: {learning_rate}  |  max_seq_length: {self.max_seq_length}")

        t0 = time.time()
        trainer = self._build_sft_trainer(
            model        = model,
            tokenizer    = tokenizer,
            train_ds     = train_ds,
            eval_ds      = eval_ds,
            output_dir   = str(output_dir / "checkpoints"),
            epochs       = epochs,
            batch_size   = batch_size,
            grad_accum   = grad_accum,
            learning_rate= learning_rate,
            warmup_steps = warmup_steps,
            logging_steps= logging_steps,
        )

        train_result = trainer.train()
        elapsed = time.time() - t0

        # Evaluar al final
        print("\n[LLMTrainer] Evaluando...")
        eval_result = trainer.evaluate()

        # Guardar adapter
        print(f"\n[LLMTrainer] Guardando adapter en {output_dir}...")
        if self._use_unsloth:
            model.save_pretrained(str(output_dir))
        else:
            trainer.model.save_pretrained(str(output_dir))

        # Guardar tokenizador (necesario para inferencia)
        tokenizer.save_pretrained(str(output_dir / "tokenizer"))

        # Métricas finales
        metrics = {
            "model_id":      self.model_id,
            "adapter_dir":   str(output_dir),
            "engine":        "unsloth" if self._use_unsloth else "peft_trl",
            "lora_r":        self.lora_r,
            "lora_alpha":    self.lora_alpha,
            "target_modules":self.target_modules,
            "epochs":        epochs,
            "batch_effective": batch_size * grad_accum,
            "train_loss":    round(train_result.training_loss, 4),
            "eval_loss":     round(eval_result.get("eval_loss", 0.0), 4),
            "train_samples": len(train_ds),
            "eval_samples":  len(eval_ds),
            "elapsed_min":   round(elapsed / 60, 2),
            "vram_peak_gb":  round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }

        # Guardar meta.json (permite checkpoint en futuros relanzamientos)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # ── Smoke test post-entrenamiento (S4.1) ─────────────────────
        print("\n[LLMTrainer] Ejecutando smoke test...")
        smoke_result = self._run_smoke_test(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_ds,
        )
        metrics["smoke_test"] = smoke_result

        # Re-guardar meta.json con smoke test
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # ── Generar informe post-entrenamiento (S4.2) ────────────────
        try:
            from motor.report import generate_training_report
            generate_training_report(
                adapter_dir=str(output_dir),
                metrics=metrics,
                smoke_test=smoke_result,
            )
        except Exception as e:
            print(f"[LLMTrainer] No se pudo generar el informe: {e}")
        # ──────────────────────────────────────────────────────────────
        # ──────────────────────────────────────────────────────────────

        print("\n" + "=" * 60)
        print("  Entrenamiento completado")
        print("=" * 60)
        print(f"  Train loss:   {metrics['train_loss']}")
        print(f"  Eval loss:    {metrics['eval_loss']}")
        print(f"  Tiempo:       {metrics['elapsed_min']} min")
        print(f"  VRAM pico:    {metrics['vram_peak_gb']} GB")
        print(f"  Adapter:      {output_dir}")
        print("=" * 60)

        return metrics

    # ------------------------------------------------------------------
    # Smoke test post-entrenamiento (S4.1)
    # ------------------------------------------------------------------

    def _run_smoke_test(
        self,
        model,
        tokenizer,
        eval_dataset,
        device: str = "cuda",
        num_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Verifica que el adapter realmente cambia el output del modelo.

        Compara inferencia base (sin LoRA) vs adapter (con LoRA) en
        los mismos ejemplos de validación. Si las respuestas son
        casi idénticas, el adapter posiblemente no se entrenó bien.

        Parámetros
        ----------
        model : PeftModel o modelo Unsloth
        tokenizer : AutoTokenizer
        eval_dataset : Dataset
        device : str
        num_samples : int

        Devuelve
        --------
        dict con:
            passed: bool
            avg_difference_pct: float — % medio de diferencia
            samples: list[dict] — ejemplos con base vs adapter
        """
        import re as _re

        # Detectar si el modelo soporta desactivar adapter (solo PeftModel)
        can_disable_adapter = hasattr(model, 'disable_adapter')

        samples = []
        total_diff = 0.0
        valid = 0

        for i in range(min(num_samples, len(eval_dataset))):
            example = eval_dataset[i]

            # Extraer mensajes del ejemplo
            if "messages" in example:
                msgs = example["messages"]
                user_msg = next((m for m in msgs if m.get("role") == "user"), None)
                assistant_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
                if not user_msg:
                    continue
                input_text = str(user_msg.get("content", ""))
                expected = str(assistant_msg.get("content", "")) if assistant_msg else ""
            elif "text" in example:
                text = example["text"]
                input_text = text
                expected = ""
            else:
                continue

            if not input_text.strip():
                continue

            # Tokenizar input
            inputs = tokenizer(input_text[:1024], return_tensors="pt").to(device)

            # Inferencia CON adapter
            with torch.no_grad():
                try:
                    # Usar generate en modo adapter (ya está cargado)
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                        or tokenizer.eos_token_id,
                    )
                    adapter_output = tokenizer.decode(
                        output_ids[0][len(inputs.input_ids[0]):],
                        skip_special_tokens=True,
                    ).strip()
                except Exception as e:
                    adapter_output = f"[ERROR: {e}]"

            # Inferencia SIN adapter (desactivar LoRA)
            # Solo si el modelo lo soporta — si no, se omite la comparación
            if not can_disable_adapter:
                # Sin disable_adapter, no podemos obtener output base.
                # El smoke test quedará como "skipped".
                base_output = ""
            else:
                with torch.no_grad():
                    try:
                        model.disable_adapter()
                        output_ids_base = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                            or tokenizer.eos_token_id,
                        )
                        base_output = tokenizer.decode(
                            output_ids_base[0][len(inputs.input_ids[0]):],
                            skip_special_tokens=True,
                        ).strip()
                        model.enable_adapter()
                    except Exception as e:
                        base_output = ""
                        try:
                            model.enable_adapter()
                        except Exception:
                            pass

            # Calcular diferencia (simple: n-gram overlap)
            def _ngrams(text: str, n: int = 3) -> set:
                words = _re.findall(r'\w+', text.lower())
                return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)}

            # Si no se pudo obtener base_output (modelo sin disable_adapter),
            # este sample no cuenta para el cálculo de diff
            if not base_output and not can_disable_adapter:
                diff = 0.0  # no aplica, se ignora en el promedio
            else:
                base_ngrams = _ngrams(base_output, 3)
                adapter_ngrams = _ngrams(adapter_output, 3)

                if base_ngrams or adapter_ngrams:
                    union = base_ngrams | adapter_ngrams
                    intersection = base_ngrams & adapter_ngrams
                    overlap = len(intersection) / len(union) if union else 0.0
                    diff = (1.0 - overlap) * 100
                else:
                    diff = 100.0 if adapter_output != base_output else 0.0

            total_diff += diff
            valid += 1

            samples.append({
                "input": input_text[:120],
                "expected": expected[:80],
                "base_output": base_output[:120],
                "adapter_output": adapter_output[:120],
                "difference_pct": round(diff, 1),
            })

        avg_diff = total_diff / valid if valid > 0 else 0.0
        passed = avg_diff >= 10.0  # al menos 10% diferente para considerar que funciona

        result = {
            "passed": passed,
            "skipped": False,
            "avg_difference_pct": round(avg_diff, 1),
            "threshold": 10.0,
            "samples": samples,
            "samples_tested": valid,
        }

        if not can_disable_adapter:
            result["passed"] = None
            result["skipped"] = True
            result["skip_reason"] = (
                "Modelo sin disable_adapter() — no es un PeftModel. "
                "El smoke test solo funciona con adapters PEFT (LoRA/QLoRA). "
                "Para modelos Unsloth o modelos base sin adapter, el test no aplica."
            )
            print(f"\n  [SMOKE TEST] ⏭ OMITIDO — {result['skip_reason']}")
        elif passed:
            print(f"\n  [SMOKE TEST] ✅ ADAPTER FUNCIONA "
                  f"(diferencia media: {avg_diff:.1f}%)")
        else:
            print(f"\n  [SMOKE TEST] ❌ ADAPTER POSIBLEMENTE NO ENTRENADO "
                  f"(diferencia media: {avg_diff:.1f}% < 10%)")
            print(f"  Verifica: carga del modelo, merge, dataset, "
                  f"target_modules")

        for s in samples:
            print(f"  Input:  {s['input'][:80]}...")
            print(f"  Base:   {s['base_output'][:80]}...")
            print(f"  Adapter:{s['adapter_output'][:80]}...")
            print(f"  Diff:   {s['difference_pct']:.1f}%")
            print()

        return result

    # ------------------------------------------------------------------
    # Rama A — Unsloth (Ampere+)
    # ------------------------------------------------------------------

    def _load_unsloth(self):
        """Carga el modelo y aplica LoRA usando Unsloth (2× más rápido)."""
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = self.model_id,
            max_seq_length = self.max_seq_length,
            dtype          = None,        # Unsloth lo detecta automáticamente
            load_in_4bit   = self.load_in_4bit,
            cache_dir      = self.cache_dir,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r                      = self.lora_r,
            target_modules         = self.target_modules,
            lora_alpha             = self.lora_alpha,
            lora_dropout           = self.lora_dropout,
            bias                   = "none",
            use_gradient_checkpointing = "unsloth",  # ahorra ~30% VRAM adicional
            random_state           = 42,
            use_rslora             = True,   # rsLoRA: escala α/√r en vez de α/r
        )                                    # → adapters más estables tras GGUF Q4

        self._print_trainable_params(model)
        return model, tokenizer

    # ------------------------------------------------------------------
    # Rama A-fallback — PEFT + TRL (cualquier GPU CUDA)
    # ------------------------------------------------------------------

    def _read_raw_config(self) -> dict:
        """
        Lee el config.json del modelo SIN pasar por AutoConfig: funciona
        aunque la versión instalada de transformers no conozca la
        arquitectura (necesario para detectar p.ej. gemma4_unified).
        """
        local = Path(self.model_id) / "config.json"
        if local.exists():
            try:
                return json.loads(local.read_text(encoding="utf-8"))
            except Exception:
                return {}
        try:
            from huggingface_hub import hf_hub_download
            fp = hf_hub_download(self.model_id, "config.json", cache_dir=self.cache_dir)
            return json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _detect_multimodal_lm(self) -> bool:
        """True si el model_type es una arquitectura multimodal unified."""
        model_type = str(self._read_raw_config().get("model_type", "")).lower()
        if model_type in _MULTIMODAL_LM_TYPES:
            print(f"  [INFO] Arquitectura multimodal unified detectada: {model_type} "
                  f"→ carga con AutoModelForMultimodalLM (rama PEFT+TRL)")
            return True
        return False

    @staticmethod
    def _filter_target_modules(model, targets: List[str]) -> List[str]:
        """
        Mantiene solo los target_modules que existen en el modelo cargado.

        Necesario para arquitecturas con proyecciones fusionadas: las capas
        globales de gemma4_unified usan KV unificado (attention_k_eq_v), así
        que algunos nombres clásicos (k_proj/v_proj) pueden no existir o
        existir solo en parte de las capas. Si ninguno coincide, autodetecta
        las proyecciones lineales del modelo.
        """
        import torch.nn as nn
        suffixes = {
            name.rsplit(".", 1)[-1]
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear)
        }
        kept    = [t for t in targets if t in suffixes]
        dropped = [t for t in targets if t not in suffixes]
        if dropped and kept:
            print(f"  [INFO] target_modules ajustados al modelo real: {kept}")
            print(f"         (sin coincidencia, descartados: {dropped})")
        if not kept:
            kept = sorted(
                s for s in suffixes
                if s.endswith("_proj") or s in ("query", "key", "value", "dense")
            )
            print(f"  [WARN] Ningún target_module coincidía con el modelo — "
                  f"autodetectados: {kept}")
        return kept or list(targets)

    def _load_peft_trl(self):
        """
        Carga el modelo con transformers (+ bitsandbytes opcional) y aplica
        LoRA con PEFT.

        Modos soportados:
          - GPU CUDA (Pascal, Volta, Ampere…): 4-bit opcional, fp16/bf16
          - CPU offload (iGPU / GPU < 4 GB):  device_map="cpu", fp32, sin 4-bit
          - CPU puro (sin GPU):               device_map="cpu", fp32, sin 4-bit

        Arquitecturas multimodales unified (gemma4_unified): carga con
        AutoProcessor + AutoModelForMultimodalLM; el resto del pipeline
        (LoRA, SFTTrainer) es idéntico — "fine-tuned in one pass".
        """
        from peft import get_peft_model, LoraConfig, TaskType

        if self._multimodal_lm:
            try:
                from transformers import AutoProcessor, AutoModelForMultimodalLM
            except ImportError as e:
                raise ImportError(
                    "Esta arquitectura multimodal requiere una versión reciente "
                    "de transformers (AutoModelForMultimodalLM no disponible).\n"
                    "Actualiza con: pip install -U transformers"
                ) from e
            _model_cls = AutoModelForMultimodalLM
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir         = self.cache_dir,
                trust_remote_code = True,
            )
            # El processor envuelve al tokenizer; el resto del pipeline
            # (SFTTrainer, datasets de texto) trabaja con el tokenizer
            tokenizer = getattr(processor, "tokenizer", processor)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            _model_cls = AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir         = self.cache_dir,
                trust_remote_code = True,
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        cpu_mode = self._use_cpu or self._cpu_offload

        # dtype objetivo
        if cpu_mode:
            _target_dtype = torch.float32
        else:
            _target_dtype = torch.bfloat16 if self._use_bf16 else torch.float16

        # Mapa de dispositivo
        device_map = "cpu" if cpu_mode else "auto"

        # Configuración de cuantización
        model_kwargs = {
            "torch_dtype":        _target_dtype,
            "device_map":         device_map,
            "cache_dir":          self.cache_dir,
            "trust_remote_code":  True,
        }

        if self.load_in_4bit and not cpu_mode:
            from motor._model_utils import apply_4bit_quantization
            apply_4bit_quantization(model_kwargs, dtype=_target_dtype)

        # Cargar modelo
        try:
            model = _model_cls.from_pretrained(self.model_id, **model_kwargs)
        except (ValueError, KeyError) as e:
            if self._multimodal_lm:
                import transformers as _tf
                raise RuntimeError(
                    f"transformers {_tf.__version__} no reconoce la arquitectura "
                    f"de '{self.model_id}'. Actualiza: pip install -U transformers"
                ) from e
            raise

        # Preparar para kbit training / gradient checkpointing
        if self.load_in_4bit and "quantization_config" in model_kwargs and not cpu_mode:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
            )
        elif self._grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Inyectar LoRA
        # use_rslora=True: Rank-Stabilized LoRA (escala α/√r en vez de α/r).
        # Hace los adapters más resistentes a la cuantización posterior (GGUF Q4_K_M),
        # reduciendo falsos rollbacks del 15% en ContinualLearner.
        lora_config = LoraConfig(
            r              = self.lora_r,
            lora_alpha     = self.lora_alpha,
            # Validados contra el modelo cargado (gemma4_unified fusiona KV
            # en capas globales → no todos los nombres clásicos existen)
            target_modules = self._filter_target_modules(model, self.target_modules),
            lora_dropout   = self.lora_dropout,
            bias           = "none",
            task_type      = TaskType.CAUSAL_LM,
            use_rslora     = True,
        )
        model = get_peft_model(model, lora_config)

        # --- FIX Pascal (compute < 8.0): bf16 no es soportado ---
        # Solo aplica en modo GPU sin bf16. En CPU usamos fp32 directamente.
        if not self._use_bf16 and not cpu_mode:
            n_cast = 0
            for param in model.parameters():
                if param.dtype == torch.bfloat16:
                    param.data = param.data.to(torch.float16)
                    n_cast += 1
            if n_cast:
                print(f"  [INFO] {n_cast} tensores bf16 → fp16 (GPU Pascal, compute < 8.0)")

        self._print_trainable_params(model)
        return model, tokenizer

    # ------------------------------------------------------------------
    # SFTTrainer — común para ambas ramas
    # ------------------------------------------------------------------

    def _build_sft_trainer(
        self,
        model,
        tokenizer,
        train_ds,
        eval_ds,
        output_dir: str,
        epochs: int,
        batch_size: int,
        grad_accum: int,
        learning_rate: float,
        warmup_steps: int,
        logging_steps: int,
    ):
        """Construye el SFTTrainer de TRL (común para Unsloth y PEFT+TRL)."""

        # Fix Windows: trl lee plantillas .jinja sin especificar encoding,
        # lo que falla con cp1252. Parcheamos pathlib.Path.read_text para que
        # use utf-8 por defecto. Debe hacerse ANTES del import de trl.
        import pathlib as _pathlib
        if not getattr(_pathlib.Path.read_text, "_utf8_patched", False):
            _orig_rt = _pathlib.Path.read_text
            def _utf8_rt(self, encoding=None, errors=None, newline=None):
                return _orig_rt(self, encoding=encoding or "utf-8",
                                errors=errors, newline=newline)
            _utf8_rt._utf8_patched = True
            _pathlib.Path.read_text = _utf8_rt

        from trl import SFTTrainer, SFTConfig
        from transformers import TrainerCallback

        # Wrapper para adaptar nuestro callback a la interfaz TrainerCallback
        class _CallbackAdapter(TrainerCallback):
            def __init__(self, cb):
                self._cb = cb
            def on_train_begin(self, args, state, control, **kw):
                self._cb.on_train_begin(args, state, control, **kw)
            def on_epoch_begin(self, args, state, control, **kw):
                self._cb.on_epoch_begin(args, state, control, **kw)
            def on_log(self, args, state, control, logs=None, **kw):
                self._cb.on_log(args, state, control, logs=logs, **kw)
            def on_train_end(self, args, state, control, **kw):
                self._cb.on_train_end(args, state, control, **kw)

        progress_cb = _CallbackAdapter(_LiveProgressCallback(total_epochs=epochs))

        import inspect as _inspect

        # max_seq_length: la forma más robusta y agnóstica a la versión de TRL
        # es fijarlo en el tokenizer. Así funciona en todas las versiones.
        tokenizer.model_max_length = self.max_seq_length

        # 'dataset_text_field' y 'packing' existen en TRL < 0.16 aprox.
        _sft_params = set(_inspect.signature(SFTConfig.__init__).parameters)
        _text_field = {"dataset_text_field": "text"} if "dataset_text_field" in _sft_params else {}
        _packing    = {"packing": False}              if "packing"            in _sft_params else {}

        training_args = SFTConfig(
            output_dir                  = output_dir,
            num_train_epochs            = epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            gradient_accumulation_steps = grad_accum,
            gradient_checkpointing      = True,
            gradient_checkpointing_kwargs = {"use_reentrant": False},
            warmup_steps                = warmup_steps,
            learning_rate               = learning_rate,
            # Pascal (compute < 8.0) + 4-bit: NO usar fp16 en el trainer.
            # bitsandbytes ya computa en fp16 via bnb_4bit_compute_dtype.
            # Si activamos fp16=True el GradScaler falla con tensores bf16
            # que gradient_checkpointing regenera. En Ampere+ usamos bf16.
            fp16                        = False,
            bf16                        = self._use_bf16,
            logging_steps               = logging_steps,
            eval_strategy               = "epoch",
            save_strategy               = "no",
            report_to                   = "none",
            seed                        = 42,
            disable_tqdm                = True,
            **_text_field,
            **_packing,
        )

        # TRL >= 0.12 renombró 'tokenizer' → 'processing_class' en SFTTrainer
        _trainer_params = set(_inspect.signature(SFTTrainer.__init__).parameters)
        _tok_kwarg = (
            {"processing_class": tokenizer}
            if "processing_class" in _trainer_params
            else {"tokenizer": tokenizer}
        )

        return SFTTrainer(
            model           = model,
            train_dataset   = train_ds,
            eval_dataset    = eval_ds,
            args            = training_args,
            callbacks       = [progress_cb],
            **_tok_kwarg,
        )

    # ------------------------------------------------------------------
    # Carga del dataset
    # ------------------------------------------------------------------

    def _load_dataset(self, dataset_path: str, eval_split: float):
        """
        Carga el .jsonl y divide en train/eval.
        Solo soporta el formato ChatML generado por DataDigestor.
        """
        from datasets import load_dataset

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"[LLMTrainer] Dataset no encontrado: {dataset_path}\n"
                "Genera el dataset primero con DataDigestor."
            )

        ds = load_dataset("json", data_files=str(path), split="train")

        # Verificar formato
        if "messages" not in ds.column_names:
            raise ValueError(
                "[LLMTrainer] El dataset no tiene campo 'messages'.\n"
                "Asegúrate de usar el formato ChatML generado por DataDigestor."
            )

        # Dividir train / eval
        split = ds.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def _print_trainable_params(model) -> None:
        """Imprime cuántos parámetros son entrenables (los del adapter LoRA)."""
        try:
            model.print_trainable_parameters()
        except AttributeError:
            total     = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            pct = trainable / total * 100
            print(f"  Parámetros entrenables: {trainable:,} / {total:,} ({pct:.3f}%)")
