"""
fabrica_loras.py
================
Fábrica de LoRAs Especializados — orquestador principal.

Uso como CLI:
    python fabrica_loras.py digestor --data titanic.csv \
        --label-col Survived \
        --label-map "0:NO,1:YES" \
        --task "¿Sobrevivió este pasajero al Titanic? Responde YES o NO." \
        --output dataset.jsonl

    python fabrica_loras.py digestor --data ./docs/ \
        --task "Resume el siguiente texto en una frase." \
        --output dataset.jsonl --format chatml

Estado de implementación
-------------------------
  Fase 2 — DataDigestor     ✅ completado (9 formatos + semáforo)
  Fase 3 — ModelAnalyzer    ✅ completado (target_modules, motor, tamaño)
  Fase 4 — LLMTrainer       ✅ validado (Rama A-fallback PEFT+TRL, fix Pascal)
  Fase 4B— VLMTrainer       ✅ implementado (Rama B, PEFT+Trainer)
  Fase 5 — ExportManager    ✅ safetensors validado / GGUF implementado
  Fase 6 — CLI completa      ✅ funcional (9 subcomandos)
  Fase 7 — Servidor API      ✅ FastAPI + UI web + agente ReAct
"""

from __future__ import annotations

import argparse
import sys
import os

# Forzar UTF-8 en Windows para todas las operaciones de archivo.
# Necesario porque trl lee plantillas Jinja sin especificar encoding
# y Python usa cp1252 por defecto en Windows → UnicodeDecodeError.
# PYTHONUTF8 debe estar en os.environ ANTES de cualquier import de trl/peft.
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Fix: WinError 6714 (transacción NTFS obsoleta) provoca OSError en
# importlib._fill_cache cuando pyarrow escanea sys.path al inicializarse.
# Python ya ignora FileNotFoundError/PermissionError en _fill_cache — aquí
# extendemos esa protección a cualquier OSError (tratamos el directorio como vacío).
try:
    import importlib.machinery as _imm_fix
    def _safe_fc(self, _orig=_imm_fix.FileFinder._fill_cache):
        try:
            _orig(self)
        except OSError as _e:
            # Solo suprimir WinError 6714 (transacción NTFS obsoleta).
            # Re-lanzar cualquier otro OSError para no interferir con sklearn, etc.
            if getattr(_e, 'winerror', None) == 6714:
                if not hasattr(self, '_path_cache'):
                    self._path_cache = set()
            else:
                raise
    _imm_fix.FileFinder._fill_cache = _safe_fc
    del _imm_fix, _safe_fc
except Exception:
    pass


# ===========================================================================
# SECCIÓN 1: COMANDO — digestor
# ===========================================================================

def _cmd_digestor(args: argparse.Namespace) -> None:
    """
    Convierte datos crudos en dataset.jsonl usando DataDigestor.
    """
    from motor.digestor import DataDigestor, detect_file_type

    # --- Parsear label_map ("0:NO,1:YES" → {0: "NO", 1: "YES"}) ---
    label_map = {}
    if args.label_map:
        for pair in args.label_map.split(","):
            pair = pair.strip()
            if ":" not in pair:
                print(f"[WARN] Par inválido en label-map (ignorado): {pair!r}")
                continue
            k, v = pair.split(":", 1)
            k = k.strip()
            v = v.strip()
            # Intentar convertir clave a int/float
            try:
                k = int(k)
            except ValueError:
                try:
                    k = float(k)
                except ValueError:
                    pass
            label_map[k] = v

    digestor = DataDigestor(
        task=args.task,
        label_col=args.label_col,
        label_map=label_map or None,
        output_format=args.format,
        skip_nulls=not args.keep_nulls,
        model_id=getattr(args, "model", None),
        domain=getattr(args, "domain", None),
    )

    # --- Detectar si --data es fichero o carpeta ---
    data_path = args.data
    ocr_mode  = getattr(args, "ocr", False)
    vlm_mode  = getattr(args, "vlm", False)

    if os.path.isdir(data_path):
        # Comprobar si la carpeta contiene SOLO imágenes (sin CSV/JSON/TXT)
        _img_exts  = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        _text_exts = {".csv", ".json", ".jsonl", ".txt", ".md", ".pdf", ".xlsx", ".xls"}
        folder_files  = list(os.scandir(data_path))
        is_image_only = folder_files and all(
            os.path.isdir(f.path) or
            os.path.splitext(f.name)[1].lower() in _img_exts
            for f in folder_files
        )

        if vlm_mode or (is_image_only and not ocr_mode):
            # Carpeta de imágenes → VLM por defecto (o si --vlm explícito)
            if not vlm_mode and is_image_only:
                print(f"[Digestor] Carpeta de imágenes detectada → modo VLM automático")
                print(f"           (usa --ocr para forzar extracción de texto con OCR)")
            else:
                print(f"[Digestor] Modo VLM forzado por --vlm")
            digestor.from_images_folder_vlm(
                folder               = data_path,
                question             = args.task,
                label_from_subfolder = True,
            )
        else:
            # Carpeta mixta o --ocr explícito → procesado normal (OCR incluido)
            digestor.from_folder(data_path)
    else:
        file_type = detect_file_type(data_path)
        if file_type == "csv":
            digestor.from_csv(data_path, sep=args.sep)
        elif file_type in ("json", "jsonl"):
            digestor.from_json(data_path, text_field=args.text_field or "text",
                               label_field=args.label_col)
        elif file_type == "txt":
            digestor.from_txt(data_path)
        elif file_type == "pdf":
            digestor.from_pdf(data_path)
        elif file_type == "docx":
            digestor.from_docx(data_path)
        elif file_type == "html":
            digestor.from_html(data_path)
        elif file_type == "audio":
            digestor.from_audio(data_path)
        elif file_type == "video":
            digestor.from_video(data_path)
        else:
            print(f"[ERROR] Tipo de archivo no soportado: {data_path}")
            sys.exit(1)

    n = digestor.to_jsonl(args.output, shuffle=not args.no_shuffle,
                          deduplicate=not getattr(args, "no_dedup", False))
    print(f"\n✓ {n} ejemplos exportados a {args.output}")


# ===========================================================================
# SECCIÓN 2: COMANDO — analyzer
# ===========================================================================

def _cmd_analyzer(args: argparse.Namespace) -> None:
    """
    Analiza un modelo HuggingFace y muestra su configuracion LoRA optima.
    Si se proporciona --data, muestra recomendaciones personalizadas.
    """
    from motor.analyzer import ModelAnalyzer

    analyzer = ModelAnalyzer(
        model_id  = args.model,
        cache_dir = args.cache_dir,
        lora_r    = args.lora_r,
        lora_alpha= args.lora_alpha,
    )
    result = analyzer.analyze()

    # ── Recomendaciones personalizadas si hay dataset ────────────────
    if args.data:
        import json as _json
        data_path = args.data
        examples = []
        total_chars = 0
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                ex = _json.loads(line.strip())
                examples.append(ex)
                # Calcular chars del ejemplo
                if "messages" in ex:
                    chars = sum(len(m.get("content", "")) for m in ex["messages"])
                else:
                    chars = len(ex.get("instruction", "")) + len(ex.get("input", "")) + len(ex.get("output", ""))
                total_chars += chars

        num_examples = len(examples)
        avg_chars = total_chars / max(num_examples, 1)
        avg_tokens = int(avg_chars / 4)  # aprox 4 chars/token

        # Contar clases
        labels: set = set()
        for ex in examples:
            if "messages" in ex:
                for m in ex["messages"]:
                    if m.get("role") == "assistant":
                        labels.add(m.get("content", "").strip())
            else:
                labels.add(ex.get("output", "").strip())

        num_classes = len(labels) if labels else 1

        rec = analyzer.recommend(
            num_examples=num_examples,
            avg_tokens=avg_tokens,
            num_classes=num_classes,
            gpu=getattr(args, "gpu", None),
        )

        print(f"\n{'='*60}")
        print(f"  RECOMENDACIONES PARA TU DATASET")
        print(f"{'='*60}")
        print(f"  Ejemplos: {num_examples}")
        print(f"  Tokens medios: {avg_tokens}")
        print(f"  Clases: {num_classes}")
        print(f"  GPU: {rec['gpu']}")
        print()
        print(f"  🎯 Hiperparametros sugeridos:")
        print(f"     r={rec['lora_r']}, alpha={rec['lora_alpha']}, "
              f"epochs={rec['epochs']}, lr={rec['learning_rate']}")
        print(f"     → {rec['reasoning']}")
        print()
        time_est = rec["time_estimate"]
        print(f"  ⏱️  Tiempo estimado:")
        print(f"     {time_est['estimated_minutes']} min "
              f"(rango: {time_est['range_min']}-{time_est['range_max']} min)")
        print(f"     Confianza: {time_est['confidence']}")
        print(f"     Benchmark mas cercano: {time_est['closest_benchmark']}")
        print()
        print(f"  💾 VRAM estimada (4-bit): {rec['vram_estimate_4bit']:.1f} GB")
        if rec["warnings"]:
            print(f"\n  ⚠️  Advertencias:")
            for w in rec["warnings"]:
                print(f"     - {w}")
        if rec["needs_augmentation"]:
            print(f"\n  💡 Este dataset se beneficiaria de data augmentation.")

        # ── S3.2: Detectar adapters relacionados ─────────────────────
        related = ModelAnalyzer.find_related_adapters(
            model_id=args.model,
            adapters_dir="adapters",
        )
        print(f"\n  🔄 Adapters existentes:")
        if related["all_adapters"]:
            for a in related["all_adapters"]:
                loss_info = f" (loss={a['train_loss']:.3f})" if a.get("train_loss") else ""
                print(f"     - {a['name']} | {a['model_id']}{loss_info}")
            print(f"     → {related['suggestion']}")
        else:
            print(f"     No hay adapters previos. Primer entrenamiento.")
        print(f"{'='*60}")

    if args.json:
        import json as _json
        print(_json.dumps(result.to_dict(), indent=2, default=str))


# ===========================================================================
# SECCIÓN 3: COMANDO — trainer / vlm / train (unificado)
# ===========================================================================

def _detect_dataset_mode(dataset_path: str) -> str:
    """
    Lee el primer ejemplo del .jsonl y devuelve 'vlm' o 'llm'.

    Criterio: si algún mensaje tiene content con {"type": "image"} → vlm.
    En cualquier otro caso → llm.
    """
    import json as _json
    try:
        with open(dataset_path, encoding="utf-8") as f:
            first = _json.loads(f.readline())
        for msg in first.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        return "vlm"
    except Exception:
        pass
    return "llm"


def _detect_model_mode(model_id: str, cache_dir=None) -> str:
    """
    Detecta si el modelo es un VLM leyendo su config.json desde HF Hub
    (sin descargar pesos). Devuelve 'vlm' o 'llm'.

    Criterio: si config tiene 'vision_config' o model_type contiene 'vl' → vlm.
    """
    try:
        from huggingface_hub import hf_hub_download
        import json as _json
        cfg_path = hf_hub_download(
            model_id,
            filename  = "config.json",
            cache_dir = cache_dir,
        )
        with open(cfg_path, encoding="utf-8") as f:
            cfg = _json.load(f)
        if "vision_config" in cfg:
            return "vlm"
        mt = cfg.get("model_type", "").lower()
        if "vl" in mt or "vision" in mt or "llava" in mt:
            return "vlm"
    except Exception:
        pass
    return "llm"


def _auto_train_params(vram_gb: float) -> tuple:
    """
    Devuelve (batch_size, grad_accum, max_seq_length) según la VRAM disponible.
    El batch efectivo siempre es 16 (batch_size × grad_accum).

    ≥ 22 GB  → batch=4, accum=4,  max_seq=2048  (RTX 4090, 3090, A100)
    ≥ 14 GB  → batch=2, accum=8,  max_seq=2048  (RTX 4080 16/17 GB)
     ≥ 8 GB  → batch=1, accum=16, max_seq=1024  (RTX 3080 10 GB, 2080 Ti)
    <  8 GB  → batch=1, accum=16, max_seq=512   (GTX 1080 Ti y anteriores)
    """
    if vram_gb >= 22:
        return 4, 4, 2048
    elif vram_gb >= 14:
        return 2, 8, 2048
    elif vram_gb >= 8:
        return 1, 16, 1024
    else:
        return 1, 16, 512


def _cmd_trainer(args: argparse.Namespace) -> None:
    """
    Entrena un adapter LoRA sobre un LLM usando el dataset.jsonl del Digestor.
    Selecciona automáticamente Rama A (Unsloth) o Rama A-fallback (PEFT+TRL).
    """
    import torch
    from motor.trainer_llm import LLMTrainer
    from motor.analyzer import ModelAnalyzer

    # --- Auto-detección de parámetros de entrenamiento según VRAM ---
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
    else:
        vram_gb = 0.0
        gpu_name = "No detectada"

    auto_bs, auto_ga, auto_msl = _auto_train_params(vram_gb)

    batch_size     = args.batch_size     if args.batch_size     is not None else auto_bs
    grad_accum     = args.grad_accum     if args.grad_accum     is not None else auto_ga
    max_seq_length = args.max_seq_length if args.max_seq_length is not None else auto_msl

    origen_bs  = "usuario" if args.batch_size     is not None else f"auto ({gpu_name}, {vram_gb:.1f} GB)"
    origen_msl = "usuario" if args.max_seq_length is not None else f"auto ({vram_gb:.1f} GB VRAM)"

    print("=" * 60)
    print("  Parámetros de entrenamiento")
    print("=" * 60)
    print(f"  GPU:              {gpu_name}")
    print(f"  VRAM:             {vram_gb:.1f} GB")
    print(f"  batch_size:       {batch_size}  [{origen_bs}]")
    print(f"  grad_accum:       {grad_accum}")
    print(f"  batch efectivo:   {batch_size * grad_accum}")
    print(f"  max_seq_length:   {max_seq_length}  [{origen_msl}]")
    print("=" * 60)

    # Si no se pasan target_modules, usar el Analyzer para detectarlos
    target_modules = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",")]
    else:
        print("[trainer] No se especificaron target_modules — ejecutando ModelAnalyzer...")
        try:
            result = ModelAnalyzer(args.model, lora_r=args.lora_r, lora_alpha=args.lora_alpha).analyze()
            target_modules = result.target_modules
        except Exception as e:
            print(f"[WARN] ModelAnalyzer falló ({e}) — usando target_modules genéricos")
            target_modules = ["q_proj", "v_proj"]

    trainer = LLMTrainer(
        model_id        = args.model,
        target_modules  = target_modules,
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        load_in_4bit    = not args.no_4bit,
        max_seq_length  = max_seq_length,
        cache_dir       = args.cache_dir,
    )

    metrics = trainer.fit(
        dataset_path  = args.data,
        output_dir    = args.output,
        epochs        = args.epochs,
        batch_size    = batch_size,
        grad_accum    = grad_accum,
        learning_rate = args.lr,
    )

    print(f"\n✓ Adapter guardado en: {args.output}")
    print(f"  Train loss: {metrics['train_loss']}  |  Eval loss: {metrics['eval_loss']}")
    print(f"  Tiempo: {metrics['elapsed_min']} min  |  VRAM pico: {metrics['vram_peak_gb']} GB")


# ===========================================================================
# ===========================================================================
# SECCIÓN 4B: COMANDO — vlm (entrena adapter LoRA sobre Vision-Language Models)
# ===========================================================================

def _cmd_vlm(args: argparse.Namespace) -> None:
    """
    Entrena un adapter LoRA sobre un VLM (Vision-Language Model).
    Usa PEFT + Trainer estándar (sin TRL, que no maneja imágenes).

    El LoRA se inyecta solo en el LLM backbone; el vision encoder queda
    congelado para ahorrar VRAM y tiempo.

    Modelos recomendados:
      Qwen/Qwen2-VL-2B-Instruct  — 11 GB VRAM en 4-bit (GTX 1080 Ti ✓)
      Qwen/Qwen2-VL-7B-Instruct  — ~11 GB bordeando GTX 1080 Ti

    Dataset: generado con DataDigestor.from_images_folder_vlm() o manualmente.
    """
    import torch
    from motor.trainer_vlm import VLMTrainer

    if torch.cuda.is_available():
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
    else:
        vram_gb  = 0.0
        gpu_name = "No detectada"

    # Parámetros auto: batch=1 siempre para VLMs (imágenes consumen VRAM)
    max_seq_length = args.max_seq_length if args.max_seq_length is not None else (
        1024 if vram_gb >= 8 else 512
    )
    batch_size = args.batch_size if args.batch_size is not None else 1
    grad_accum = args.grad_accum if args.grad_accum is not None else 16

    print("=" * 60)
    print("  Parámetros VLM")
    print("=" * 60)
    print(f"  GPU:            {gpu_name}")
    print(f"  VRAM:           {vram_gb:.1f} GB")
    print(f"  batch_size:     {batch_size}  (VLMs: siempre 1 recomendado)")
    print(f"  grad_accum:     {grad_accum}  (batch efectivo: {batch_size * grad_accum})")
    print(f"  max_seq_length: {max_seq_length}")
    print("=" * 60)

    target_modules = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",")]

    trainer = VLMTrainer(
        model_id        = args.model,
        target_modules  = target_modules,
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        load_in_4bit    = not args.no_4bit,
        max_seq_length  = max_seq_length,
        cache_dir       = args.cache_dir,
    )

    metrics = trainer.fit(
        dataset_path  = args.data,
        output_dir    = args.output,
        epochs        = args.epochs,
        batch_size    = batch_size,
        grad_accum    = grad_accum,
        learning_rate = args.lr,
    )

    print(f"\n✓ Adapter VLM guardado en: {args.output}")
    print(f"  Train loss: {metrics['train_loss']}  |  Eval loss: {metrics['eval_loss']}")
    print(f"  Tiempo: {metrics['elapsed_min']} min  |  VRAM pico: {metrics['vram_peak_gb']} GB")


# ===========================================================================
# SECCIÓN 4C: COMANDO — train (unificado, auto-detecta LLM vs VLM)
# ===========================================================================

def _cmd_train(args: argparse.Namespace) -> None:
    """
    Comando unificado que detecta automáticamente si debe usar LLMTrainer o
    VLMTrainer, sin que el usuario tenga que saber cuál usar.

    Lógica de auto-detección (en orden de prioridad):
      1. --mode llm|vlm  → el usuario lo fuerza explícitamente
      2. Dataset         → si tiene {"type": "image"} en mensajes → vlm
      3. Modelo          → si config.json tiene vision_config → vlm
      4. Fallback        → llm
    """
    mode = getattr(args, "mode", "auto")

    if mode == "auto":
        # 1. Intentar detectar por dataset
        detected_ds = _detect_dataset_mode(args.data)
        if detected_ds == "vlm":
            mode = "vlm"
            print(f"[train] Auto-detectado: dataset MULTIMODAL → usando VLMTrainer")
        else:
            # 2. Intentar detectar por modelo
            print(f"[train] Analizando arquitectura del modelo: {args.model}")
            detected_model = _detect_model_mode(args.model, cache_dir=args.cache_dir)
            mode = detected_model
            if mode == "vlm":
                print(f"[train] Auto-detectado: modelo VLM (vision_config presente) → usando VLMTrainer")
            else:
                print(f"[train] Auto-detectado: modelo LLM → usando LLMTrainer")
    else:
        print(f"[train] Modo forzado por usuario: {mode.upper()}Trainer")

    # Delegar al comando específico reutilizando su lógica
    if mode == "vlm":
        _cmd_vlm(args)
    else:
        _cmd_trainer(args)


# ===========================================================================
# SECCIÓN 4D: COMANDO — export (fusiona adapter LoRA y exporta a safetensors/GGUF)
# ===========================================================================

def _cmd_export(args: argparse.Namespace) -> None:
    """
    Fusiona un adapter LoRA con su modelo base y exporta el resultado.
    Soporta dos formatos:
      safetensors — modelo HuggingFace completo (default)
      gguf        — modelo para llama.cpp / Ollama / LM Studio (CPU, sin GPU)
    """
    from motor.exporter import ExportManager

    em = ExportManager(
        adapter_dir = args.adapter,
        base_model  = args.model,
        cache_dir   = args.cache_dir,
    )

    fmt = args.format.lower()
    if fmt == "safetensors":
        out = em.to_safetensors(args.output)
    elif fmt == "gguf":
        out = em.to_gguf(
            args.output,
            quantization = args.quantization,
            merged_dir   = args.merged_dir,
        )
    else:
        print(f"[ERROR] Formato desconocido: {fmt}. Usa 'safetensors' o 'gguf'.")
        raise SystemExit(1)

    print(f"\n✓ Export completado: {out}")


# ===========================================================================
# ===========================================================================
# SECCIÓN 5B: COMANDO — chat GGUF (CPU, llama-cpp-python)
# ===========================================================================

def _cmd_chat_gguf(args: argparse.Namespace) -> None:
    """
    Chat interactivo con un modelo GGUF usando llama-cpp-python.
    Corre en CPU puro, sin GPU ni CUDA. Requiere solo:
      pip install llama-cpp-python
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print("\n[ERROR] Para usar archivos .gguf necesitas llama-cpp-python.")
        print("        Instálalo con:")
        print("          pip install llama-cpp-python")
        raise SystemExit(1)

    import os

    model_path = args.model
    print(f"\n[Chat GGUF] Cargando: {model_path}")
    print("            Usando CPU. Primera carga puede tardar 20-40s...")

    llm = Llama(
        model_path = model_path,
        n_ctx      = 4096,
        n_threads  = os.cpu_count() or 4,
        verbose    = False,
    )

    system_prompt = args.system or "Eres un asistente útil. Responde de forma clara y concisa."
    historial: list[dict] = []

    print(f"\n{'='*60}")
    print("  Chat GGUF listo (CPU). Comandos especiales:")
    print("    /reset           — borrar historial")
    print("    /system <texto>  — cambiar prompt de sistema")
    print("    /quit            — salir")
    print(f"{'='*60}\n")

    while True:
        try:
            entrada = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chat] Sesión terminada.")
            break

        if not entrada:
            continue

        if entrada == "/quit":
            print("[Chat] Sesión terminada.")
            break
        elif entrada == "/reset":
            historial = []
            print("[Chat] Historial borrado.\n")
            continue
        elif entrada.startswith("/system "):
            system_prompt = entrada[8:].strip()
            historial = []
            print("[Chat] System prompt actualizado. Historial borrado.\n")
            continue

        historial.append({"role": "user", "content": entrada})
        messages = [{"role": "system", "content": system_prompt}] + historial

        result = llm.create_chat_completion(
            messages    = messages,
            max_tokens  = args.max_tokens,
            temperature = args.temperature if args.temperature > 0 else 1.0,
            top_p       = args.top_p,
        )
        respuesta = result["choices"][0]["message"]["content"].strip()
        historial.append({"role": "assistant", "content": respuesta})
        print(f"\nIA: {respuesta}\n")


# ===========================================================================
# ===========================================================================
# SECCIÓN 5: COMANDO — chat (inferencia interactiva con un adapter o modelo)
# ===========================================================================

def _cmd_chat(args: argparse.Namespace) -> None:
    """
    Chat interactivo con un modelo o adapter LoRA desde la línea de comandos.
    Soporta:
      - Archivo GGUF (.gguf)  → usa llama-cpp-python, CPU puro, sin GPU
      - Adapter LoRA (carpeta con adapter_config.json + meta.json)
      - Modelo fusionado en safetensors (carpeta HuggingFace estándar)
      - Modelo HuggingFace por ID (Qwen/Qwen2.5-3B-Instruct, etc.)
    """
    # Detectar si es un archivo GGUF — delega a _cmd_chat_gguf
    if args.model.lower().endswith(".gguf"):
        _cmd_chat_gguf(args)
        return

    # Pre-importar dependencias transitivas ANTES de que transformers las cargue
    # via su sistema de lazy imports (que puede fallar con import lock anidado
    # o con errores de filesystem heredados del proceso de entrenamiento).
    try:
        import sklearn          # noqa: F401
        import sklearn.utils    # noqa: F401
    except Exception:
        pass
    try:
        import pyarrow          # noqa: F401
    except Exception:
        pass

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model
    is_adapter  = _is_adapter_dir(model_path)

    print(f"\n[Chat] Cargando {'adapter LoRA' if is_adapter else 'modelo'}...")
    print(f"       Ruta: {model_path}")

    # --- Detectar dtype según GPU y decidir estrategia de carga ---
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"       GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    else:
        dtype = torch.float32
        vram_gb = 0.0
        gpu_name = None
        print("       Modo CPU (sin GPU detectada — puede ser lento)")

    # Usar cuantización 4-bit cuando el modelo podría no caber en VRAM.
    # Qwen2.5-14B en bf16 ocupa ~28 GB; en 4-bit baja a ~8 GB.
    # Esto además evita el disk-offload que causa KeyError en PEFT _update_offload.
    use_4bit = vram_gb > 0 and vram_gb < 22
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            # llm_int8_enable_fp32_cpu_offload=True permite que los layers que no
            # caben en VRAM (ej. 14B en 1080 Ti 11 GB) se ejecuten en RAM CPU.
            # Resultado: modelo carga correctamente aunque sea lento en esos layers.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit                   = True,
                bnb_4bit_compute_dtype         = dtype,
                bnb_4bit_use_double_quant      = True,
                bnb_4bit_quant_type            = "nf4",
                llm_int8_enable_fp32_cpu_offload = True,
            )
            model_kwargs = dict(
                quantization_config = bnb_config,
                device_map          = "auto",
            )
            print(f"       Modo: 4-bit NF4 (GPU+CPU offload si es necesario)")
        except ImportError:
            use_4bit = False

    if not use_4bit:
        model_kwargs = dict(
            dtype      = dtype,
            device_map = "auto" if vram_gb > 0 else "cpu",
        )
        print(f"       Modo: {'bf16' if dtype == torch.bfloat16 else 'fp16'}")

    # --- Cargar modelo base ---
    if is_adapter:
        # Leer modelo base del meta.json
        import json
        from pathlib import Path
        meta_path = Path(model_path) / "meta.json"
        if not meta_path.exists():
            print("[ERROR] No se encontró meta.json en el adapter. Usa --base-model para indicar el modelo base.")
            raise SystemExit(1)
        with open(meta_path) as f:
            meta = json.load(f)
        base_model_id = args.base_model or meta.get("model_id") or meta.get("base_model")
        if not base_model_id:
            print("[ERROR] meta.json no contiene 'model_id'. Usa --base-model.")
            raise SystemExit(1)
        print(f"       Base: {base_model_id}")
    else:
        base_model_id = model_path

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir         = args.cache_dir,
        trust_remote_code = True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir         = args.cache_dir,
        trust_remote_code = True,
        **model_kwargs,
    )

    # --- Cargar adapter si corresponde ---
    if is_adapter:
        from peft import PeftModel
        print("[Chat] Cargando adapter LoRA sobre el modelo base...")
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)

    model.eval()

    # --- Prompt de sistema ---
    system_prompt = args.system or "Eres un asistente útil. Responde de forma clara y concisa."

    # --- Bucle de chat ---
    historial = []
    print(f"\n{'='*60}")
    print("  Chat listo. Escribe tu mensaje. Comandos especiales:")
    print("    /reset  — borrar historial de conversación")
    print("    /system — cambiar prompt de sistema")
    print("    /quit   — salir")
    print(f"{'='*60}\n")

    while True:
        try:
            entrada = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chat] Sesión terminada.")
            break

        if not entrada:
            continue

        # Comandos especiales
        if entrada == "/quit":
            print("[Chat] Sesión terminada.")
            break
        elif entrada == "/reset":
            historial = []
            print("[Chat] Historial borrado.\n")
            continue
        elif entrada.startswith("/system "):
            system_prompt = entrada[8:].strip()
            historial = []
            print(f"[Chat] Prompt de sistema actualizado. Historial borrado.\n")
            continue

        # Añadir mensaje del usuario al historial
        historial.append({"role": "user", "content": entrada})

        # Construir prompt con template del modelo
        mensajes = [{"role": "system", "content": system_prompt}] + historial
        text = tokenizer.apply_chat_template(
            mensajes,
            tokenize            = False,
            add_generation_prompt = True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generar respuesta
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens  = args.max_tokens,
                do_sample       = args.temperature > 0,
                temperature     = args.temperature if args.temperature > 0 else 1.0,
                top_p           = args.top_p,
                repetition_penalty = 1.1,
                pad_token_id    = tokenizer.eos_token_id,
            )

        respuesta = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens = True,
        ).strip()

        # Añadir respuesta al historial
        historial.append({"role": "assistant", "content": respuesta})

        print(f"\nIA: {respuesta}\n")


def _is_adapter_dir(path: str) -> bool:
    """Detecta si una ruta es un adapter LoRA o un modelo completo."""
    from pathlib import Path
    p = Path(path)
    return (p / "adapter_config.json").exists()


# ===========================================================================
# SECCIÓN 6: COMANDO — serve (API REST FastAPI)
# ===========================================================================

def _cmd_serve(args: argparse.Namespace) -> None:
    """
    Arranca el servidor REST FastAPI con el modelo o adapter indicado.
    El modelo se carga UNA vez y queda en memoria para responder en ms.

    Endpoints disponibles:
      GET  /health         → estado, GPU, VRAM, modelo cargado
      POST /chat           → inferencia stateless (sin historial)
      POST /chat/session   → inferencia con historial de sesión
      DELETE /chat/session → borrar historial de sesión
      GET  /docs           → documentación interactiva (Swagger UI)

    Dependencias necesarias en el servidor:
      pip install fastapi uvicorn
    """
    from motor.server import run_server

    run_server(
        model_path = args.model,
        host       = args.host,
        port       = args.port,
        base_model = args.base_model,
        cache_dir  = args.cache_dir,
        api_key    = args.api_key,
        ui_only    = getattr(args, "ui_only", False),
    )


# ===========================================================================
# SECCIÓN 7: COMANDO — info (inspecciona un dataset.jsonl)
# ===========================================================================
# ===========================================================================

def _cmd_info(args: argparse.Namespace) -> None:
    """
    Muestra información sobre un dataset.jsonl existente.
    """
    import json
    from pathlib import Path

    path = Path(args.file)
    if not path.exists():
        print(f"[ERROR] Archivo no encontrado: {path}")
        sys.exit(1)

    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n{'='*55}")
    print(f"  Dataset: {path.name}")
    print(f"  Ejemplos totales: {len(lines)}")

    if lines:
        first = json.loads(lines[0])
        print(f"  Formato detectado: ", end="")
        if "messages" in first:
            print("ChatML")
            roles = [m["role"] for m in first["messages"]]
            print(f"  Roles en primer ejemplo: {roles}")
            # Mostrar primer ejemplo completo
            print(f"\n  --- Primer ejemplo ---")
            for m in first["messages"]:
                role = m["role"].upper()
                content = m["content"]
                if len(content) > 120:
                    content = content[:120] + "..."
                print(f"  [{role}] {content}")
        elif "instruction" in first:
            print("Alpaca")
            print(f"\n  --- Primer ejemplo ---")
            for k in ("instruction", "input", "output"):
                v = str(first.get(k, ""))
                if len(v) > 120:
                    v = v[:120] + "..."
                print(f"  {k.upper()}: {v}")

        # Distribución de etiquetas
        label_counts: dict = {}
        for line in lines:
            ex = json.loads(line)
            if "messages" in ex:
                msgs = ex["messages"]
                assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "<sin etiqueta>")
                lbl = assistant
            else:
                lbl = ex.get("output") or "<sin etiqueta>"
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        print(f"\n  Distribución de etiquetas:")
        for lbl, cnt in sorted(label_counts.items()):
            pct = cnt / len(lines) * 100
            print(f"    {lbl:25s}: {cnt:5d} ({pct:.1f}%)")

    print(f"{'='*55}\n")


# ===========================================================================
# SECCIÓN 8: COMANDO — convert-dataset (G2: export universal datasets)
# ===========================================================================

def _cmd_convert_dataset(args: argparse.Namespace) -> None:
    """
    Convierte un dataset JSONL al formato de otro framework de entrenamiento.

    Frameworks soportados:
      llamafactory  → ShareGPT JSON array + dataset_info.json
      unsloth       → Alpaca JSONL (instruction / input / output)
      axolotl       → ShareGPT JSONL + axolotl_config.yml
    """
    from motor.digestor import DataDigestor

    d = DataDigestor(task="convert")
    d.load_jsonl(args.input)

    framework = args.framework.lower()
    name = args.name

    if framework == "llamafactory":
        n = d.to_llamafactory(args.output, dataset_name=name)
        print(f"\n✓ {n} ejemplos exportados a {args.output}/")
        print(f"  → Copia {name}.json y dataset_info.json a la carpeta data/ de LLaMA-Factory")
        print(f"  → Añade '{name}' al campo 'dataset:' en tu config YAML de LLaMA-Factory")

    elif framework == "unsloth":
        n = d.to_unsloth(args.output)
        print(f"\n✓ {n} ejemplos exportados a {args.output}")
        print(f"  → Carga con: dataset = load_dataset('json', data_files='{args.output}')")

    elif framework == "axolotl":
        n = d.to_axolotl(args.output, dataset_name=name)
        print(f"\n✓ {n} ejemplos exportados a {args.output}/")
        print(f"  → Edita axolotl_config.yml (model, output_dir) y lanza:")
        print(f"     accelerate launch -m axolotl.cli.train axolotl_config.yml")

    else:
        print(f"[ERROR] Framework desconocido: {args.framework!r}")
        sys.exit(1)


# ===========================================================================
# SECCIÓN 9: COMANDO — learn (S10.3: aprendizaje continuo)
# ===========================================================================

def _cmd_learn(args: argparse.Namespace) -> None:
    """
    Re-entrena un adapter LoRA con nuevos datos usando ContinualLearner.

    El learner mezcla automáticamente el nuevo dataset con un replay buffer
    de ejemplos históricos (replay_buffer_size) para evitar olvido catastrófico.
    Si el eval_loss sube más de rollback_threshold, hace rollback al adapter previo.

    Uso:
        python fabrica_loras.py learn \\
            --adapter adapters/domestic_base/ \\
            --data datasets/dataset_domestic_v2.jsonl \\
            --output adapters/domestic_v3/

        # Modo automático: usa el log de interacciones como nuevos datos
        python fabrica_loras.py learn --auto \\
            --adapter adapters/domestic_base/ \\
            --log logs/interaction_log.jsonl \\
            --output adapters/domestic_v3/
    """
    from motor.continual import ContinualLearner

    adapter_dir = args.adapter
    output_dir  = args.output or (adapter_dir.rstrip("/\\") + "_retrained")

    # --- Modo --scheduled: comprobar si se debe disparar el ciclo ahora ---
    if getattr(args, "scheduled", False):
        if not args.auto or not args.log:
            print("[ERROR] --scheduled requiere --auto y --log.")
            sys.exit(1)
        import json as _sj
        import os  as _sos
        from pathlib import Path as _SPath
        from datetime import datetime as _dt, timezone as _tz

        cycle_log_path = _SPath(output_dir.rstrip("/\\") + "_cycle_log.jsonl")
        min_new   = getattr(args, "min_new_examples", 50)
        max_days  = getattr(args, "max_days_between", 30.0)
        log_path  = args.log

        # Leer el último ciclo registrado
        last_cycle_ts: float = 0.0
        last_seen_count: int = 0
        if cycle_log_path.exists():
            try:
                entries = [
                    _sj.loads(l)
                    for l in cycle_log_path.read_text(encoding="utf-8").splitlines()
                    if l.strip()
                ]
                if entries:
                    last = entries[-1]
                    last_cycle_ts    = last.get("timestamp_epoch", 0.0)
                    last_seen_count  = last.get("examples_seen", 0)
            except Exception:
                pass

        # Contar ejemplos positivos disponibles en el log
        current_positive = 0
        if _sos.path.exists(log_path):
            try:
                with open(log_path, encoding="utf-8") as _lf:
                    for _line in _lf:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _e = _sj.loads(_line)
                            _fb = _e.get("feedback")
                            if _fb is None or _fb == 1:
                                if _e.get("user_msg") and _e.get("assistant"):
                                    current_positive += 1
                        except Exception:
                            pass
            except FileNotFoundError:
                pass

        new_examples = current_positive - last_seen_count
        now_epoch    = _dt.now(_tz.utc).timestamp()
        days_since   = (now_epoch - last_cycle_ts) / 86400.0 if last_cycle_ts else float("inf")

        trigger_reason = None
        if new_examples >= min_new:
            trigger_reason = f"{new_examples} ejemplos nuevos >= umbral {min_new}"
        elif days_since >= max_days:
            trigger_reason = f"{days_since:.1f} días sin ciclo >= máximo {max_days}"

        if trigger_reason is None:
            print(
                f"[learn --scheduled] Sin disparar.\n"
                f"  Ejemplos nuevos desde último ciclo: {new_examples} / {min_new} requeridos\n"
                f"  Días desde último ciclo            : {days_since:.1f} / {max_days} máximos\n"
                f"  Próximo ciclo en: "
                f"{max(0, min_new - new_examples)} ejemplos más "
                f"o {max(0, max_days - days_since):.1f} días."
            )
            sys.exit(0)

        print(
            f"[learn --scheduled] DISPARANDO ciclo de reentrenamiento.\n"
            f"  Motivo: {trigger_reason}"
        )
        # El ciclo sigue con la lógica normal de --auto abajo.
        # Al finalizar, registramos en el cycle_log.

    # --- Modo --auto: extraer ejemplos aceptados del log de interacciones ---
    dataset_path = args.data
    if args.auto:
        if not args.log:
            print("[ERROR] --auto requiere --log <ruta_al_interaction_log.jsonl>")
            sys.exit(1)
        import tempfile as _tmpf
        from motor.log_quality import load_sft_examples, format_report
        log_path = args.log
        print(f"[learn] Extrayendo ejemplos con feedback positivo de: {log_path}")
        # Filtro de calidad: descarta vacíos/basura/truncados/tool-calls y
        # deduplica (el doble-envío de Odysseus dejaba pares basura en el log).
        _report: dict = {}
        try:
            accepted = load_sft_examples(log_path, include_unrated=True, report=_report)
        except FileNotFoundError:
            print(f"[ERROR] Log no encontrado: {log_path}")
            sys.exit(1)
        print(format_report(_report))

        if not accepted:
            print("[learn] Sin ejemplos suficientes en el log. Nada que aprender.")
            sys.exit(0)

        print(f"[learn] {len(accepted)} ejemplos limpios extraídos del log "
              f"(feedback positivo + sin feedback)")

        # Guardar dataset temporal
        _tmp = _tmpf.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for ex in accepted:
            _tmp.write(__import__("json").dumps(ex, ensure_ascii=False) + "\n")
        _tmp.close()
        dataset_path = _tmp.name
        print(f"[learn] Dataset temporal: {dataset_path} ({len(accepted)} ejemplos)")

    if not dataset_path:
        print("[ERROR] Especifica --data <dataset.jsonl> o usa --auto --log <log.jsonl>")
        sys.exit(1)

    # --- Cargar meta.json para obtener model_id ---
    import json as _json
    from pathlib import Path as _Path

    meta_path = _Path(adapter_dir) / "meta.json"
    if not meta_path.exists():
        print(f"[ERROR] meta.json no encontrado en {adapter_dir}. "
              "Usa --base-model para indicar el modelo base.")
        sys.exit(1)
    with open(meta_path, encoding="utf-8") as _mf:
        meta = _json.load(_mf)
    model_id = args.base_model or meta.get("model_id") or meta.get("base_model")
    if not model_id:
        print("[ERROR] No se pudo determinar el modelo base desde meta.json. "
              "Usa --base-model para indicarlo.")
        sys.exit(1)

    print(f"\n[learn] Aprendizaje continuo")
    print(f"  Modelo base : {model_id}")
    print(f"  Adapter     : {adapter_dir}")
    print(f"  Dataset     : {dataset_path}")
    print(f"  Salida      : {output_dir}")
    print(f"  Replay buf  : {args.replay_buffer}")
    print(f"  Rollback    : {args.rollback_threshold * 100:.0f}% regresión\n")

    cl = ContinualLearner(
        model_id            = model_id,
        replay_buffer_size  = args.replay_buffer,
        rollback_threshold  = args.rollback_threshold,
        # Pasar kwargs del adapter base al trainer
        lora_r              = args.lora_r,
        lora_alpha          = args.lora_alpha,
    )

    # Registrar adapter base en el registry para que sirva de baseline
    cl.register_existing(adapter_dir, name=_Path(adapter_dir).name)

    # Re-entrenar
    metrics = cl.fit(
        dataset_path  = dataset_path,
        output_dir    = output_dir,
        adapter_name  = _Path(output_dir).name,
        epochs        = args.epochs,
        batch_size    = args.batch_size or 4,
        learning_rate = args.lr,
    )

    # Limpiar dataset temporal si se creó desde --auto
    if args.auto and dataset_path != args.data:
        try:
            import os as _os
            _os.unlink(dataset_path)
        except Exception:
            pass

    if metrics:
        print(f"\n[learn] ✅ Adapter actualizado en {output_dir}")
        print(f"  eval_loss     : {metrics.get('eval_loss', 'N/A')}")
        print(f"  token_accuracy: {metrics.get('token_accuracy', 'N/A')}")

    # --- Registrar ciclo en el cycle_log si modo --scheduled ---
    if getattr(args, "scheduled", False):
        import json as _cj
        from pathlib import Path as _CPath
        from datetime import datetime as _cdt, timezone as _ctz

        cycle_log_path = _CPath(output_dir.rstrip("/\\") + "_cycle_log.jsonl")
        cycle_entry = {
            "timestamp":       _cdt.now(_ctz.utc).isoformat(),
            "timestamp_epoch": _cdt.now(_ctz.utc).timestamp(),
            "trigger_reason":  trigger_reason,
            "examples_seen":   current_positive,
            "examples_new":    new_examples,
            "output_dir":      str(output_dir),
            "eval_loss":       metrics.get("eval_loss") if metrics else None,
            "token_accuracy":  metrics.get("token_accuracy") if metrics else None,
            "rollback":        (metrics or {}).get("rollback", False),
        }
        with open(cycle_log_path, "a", encoding="utf-8") as _clf:
            _clf.write(_cj.dumps(cycle_entry, ensure_ascii=False) + "\n")
        print(f"[learn --scheduled] Ciclo registrado en {cycle_log_path}")

    # Instrucciones de uso
    print(f"\n  Próximo paso: usa el adapter actualizado:")
    print(f"  python fabrica_loras.py chat --model {output_dir}")


# ===========================================================================
# SECCIÓN 3: CLI principal
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fabrica_loras",
        description="Fábrica de LoRAs Especializados — orquestador principal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Subcomando: digestor ---
    p_dig = subparsers.add_parser(
        "digestor",
        help="Convierte datos crudos en dataset.jsonl para fine-tuning LoRA",
    )
    p_dig.add_argument("--data",       required=True,  help="Ruta al archivo o carpeta de datos")
    p_dig.add_argument("--task",       required=True,  help="Descripción de la tarea para el LLM")
    p_dig.add_argument("--output",     required=True,  help="Ruta de salida del .jsonl")
    p_dig.add_argument("--label-col",  default=None,   help="Columna/campo de etiqueta")
    p_dig.add_argument("--label-map",  default=None,
                       help="Mapa de etiquetas: '0:NO,1:YES' o 'POSITIVE:POS,NEGATIVE:NEG'")
    p_dig.add_argument("--format",     default="chatml", choices=["chatml", "alpaca"],
                       help="Formato de salida (default: chatml)")
    p_dig.add_argument("--sep",        default=",",    help="Separador CSV (default: ',')")
    p_dig.add_argument("--text-field", default=None,   help="Campo de texto en JSON (default: 'text')")
    p_dig.add_argument("--keep-nulls", action="store_true",
                       help="Incluir filas con etiqueta nula (por defecto se omiten)")
    p_dig.add_argument("--no-shuffle", action="store_true",
                       help="No mezclar ejemplos al exportar")
    p_dig.add_argument("--no-dedup", action="store_true",
                       help="No deduplicar ejemplos antes de exportar "
                            "(por defecto se eliminan duplicados exactos y near-dupes)")
    p_dig.add_argument("--vlm", action="store_true",
                       help="Forzar modo VLM aunque la carpeta tenga contenido mixto. "
                            "Por defecto, una carpeta de SOLO imágenes ya activa modo VLM automáticamente.")
    p_dig.add_argument("--ocr", action="store_true",
                       help="Forzar OCR sobre imágenes en lugar de modo VLM. "
                            "Útil si tienes una carpeta de imágenes pero quieres dataset de texto.")
    p_dig.add_argument("--domain", default=None,
                       choices=["auto", "financial", "medical", "legal", "technical",
                                "conversational", "general"],
                       help="Dominio del dataset. 'auto' detecta automáticamente (default: auto). "
                            "Usa un dominio específico para forzarlo manualmente.")
    p_dig.add_argument("--model", default=None,
                       help="ID del modelo HuggingFace objetivo (opcional). "
                            "Ej: Qwen/Qwen2.5-14B-Instruct. "
                            "Si se proporciona, el Digestor adapta el formato de salida "
                            "al chat template y capacidades de ese modelo.")
    p_dig.set_defaults(func=_cmd_digestor)

    # --- Subcomando: trainer ---
    p_tr = subparsers.add_parser(
        "trainer",
        help="Entrena un adapter LoRA sobre un LLM con el dataset.jsonl del Digestor",
    )
    p_tr.add_argument("--model",          required=True,  help="ID del modelo HF. Ej: Qwen/Qwen2.5-3B-Instruct")
    p_tr.add_argument("--data",           required=True,  help="Ruta al dataset.jsonl generado por el Digestor")
    p_tr.add_argument("--output",         required=True,  help="Carpeta de salida del adapter")
    p_tr.add_argument("--epochs",         type=int,   default=3,     help="Épocas de entrenamiento (default: 3)")
    p_tr.add_argument("--batch-size",     type=int,   default=None,  help="Batch por GPU (auto si no se indica, basado en VRAM disponible)")
    p_tr.add_argument("--grad-accum",     type=int,   default=None,  help="Acumulación de gradiente (auto si no se indica)")
    p_tr.add_argument("--lr",             type=float, default=2e-4,  help="Learning rate (default: 2e-4)")
    p_tr.add_argument("--max-seq-length", type=int,   default=None,  help="Longitud máx. secuencia en tokens (auto si no se indica, basado en VRAM)")
    p_tr.add_argument("--lora-r",         type=int,   default=32,    help="Rango LoRA (default: 32)")
    p_tr.add_argument("--lora-alpha",     type=int,   default=64,    help="Alpha LoRA (default: 64 = 2x lora_r)")
    p_tr.add_argument("--target-modules", default=None,
                      help="Capas LoRA separadas por coma. Si no se indica, las detecta el Analyzer.")
    p_tr.add_argument("--no-4bit",        action="store_true", help="Desactivar cuantización 4-bit")
    p_tr.add_argument("--cache-dir",      default=None, help="Directorio de caché HF (opcional)")
    p_tr.set_defaults(func=_cmd_trainer)

    # --- Subcomando: vlm (alias explícito para VLMs) ---
    p_vlm = subparsers.add_parser(
        "vlm",
        help="Entrena un adapter LoRA sobre un VLM (Vision-Language Model)",
    )
    p_vlm.add_argument("--model",          required=True,  help="ID del VLM HF. Ej: Qwen/Qwen2-VL-2B-Instruct")
    p_vlm.add_argument("--data",           required=True,  help="Ruta al dataset.jsonl multimodal (generado por DataDigestor.from_images_folder_vlm)")
    p_vlm.add_argument("--output",         required=True,  help="Carpeta de salida del adapter")
    p_vlm.add_argument("--epochs",         type=int,   default=3,    help="Épocas (default: 3)")
    p_vlm.add_argument("--batch-size",     type=int,   default=None, help="Batch por GPU (default: 1, imágenes consumen VRAM)")
    p_vlm.add_argument("--grad-accum",     type=int,   default=None, help="Acumulación de gradiente (default: 16)")
    p_vlm.add_argument("--lr",             type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p_vlm.add_argument("--max-seq-length", type=int,   default=None, dest="max_seq_length",
                       help="Máx. tokens (default: 1024). Imagen + texto puede ser largo.")
    p_vlm.add_argument("--lora-r",         type=int,   default=16,   help="Rango LoRA (default: 16)")
    p_vlm.add_argument("--lora-alpha",     type=int,   default=32,   help="Alpha LoRA (default: 32)")
    p_vlm.add_argument("--target-modules", default=None,
                       help="Capas LoRA del LLM backbone separadas por coma.")
    p_vlm.add_argument("--no-4bit",        action="store_true", help="Desactivar cuantización 4-bit")
    p_vlm.add_argument("--cache-dir",      default=None, help="Directorio de caché HF")
    p_vlm.set_defaults(func=_cmd_vlm)

    # --- Subcomando: train (UNIFICADO — detecta LLM vs VLM automáticamente) ---
    p_train = subparsers.add_parser(
        "train",
        help="[RECOMENDADO] Entrena un adapter LoRA — detecta automáticamente si el modelo/dataset es LLM o VLM",
    )
    p_train.add_argument("--model",          required=True,  help="ID del modelo HF. Ej: Qwen/Qwen2.5-7B-Instruct o Qwen/Qwen2-VL-2B-Instruct")
    p_train.add_argument("--data",           required=True,  help="Ruta al dataset.jsonl (texto o multimodal — se auto-detecta)")
    p_train.add_argument("--output",         required=True,  help="Carpeta de salida del adapter")
    p_train.add_argument("--mode",           default="auto", choices=["auto", "llm", "vlm"],
                         help="Forzar modo: 'auto' (default), 'llm' o 'vlm'")
    p_train.add_argument("--epochs",         type=int,   default=3,    help="Épocas (default: 3)")
    p_train.add_argument("--batch-size",     type=int,   default=None, dest="batch_size",
                         help="Batch por GPU (auto según VRAM y modo)")
    p_train.add_argument("--grad-accum",     type=int,   default=None, dest="grad_accum",
                         help="Acumulación de gradiente (auto)")
    p_train.add_argument("--lr",             type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p_train.add_argument("--max-seq-length", type=int,   default=None, dest="max_seq_length",
                         help="Máx. tokens (auto según VRAM)")
    p_train.add_argument("--lora-r",         type=int,   default=16,   dest="lora_r",    help="Rango LoRA (default: 16)")
    p_train.add_argument("--lora-alpha",     type=int,   default=32,   dest="lora_alpha", help="Alpha LoRA (default: 32)")
    p_train.add_argument("--target-modules", default=None, dest="target_modules",
                         help="Capas LoRA separadas por coma. Si no se indica, se detectan automáticamente.")
    p_train.add_argument("--no-4bit",        action="store_true", dest="no_4bit", help="Desactivar cuantización 4-bit")
    p_train.add_argument("--cache-dir",      default=None, dest="cache_dir", help="Directorio de caché HF")
    p_train.set_defaults(func=_cmd_train)

    # --- Subcomando: chat ---
    p_ch = subparsers.add_parser(
        "chat",
        help="Chat interactivo con un adapter LoRA o modelo HuggingFace",
    )
    p_ch.add_argument("--model",       required=True,  help="Ruta al adapter, modelo fusionado o ID HuggingFace")
    p_ch.add_argument("--base-model",  default=None,   dest="base_model",
                      help="ID del modelo base (solo necesario si meta.json no lo contiene)")
    p_ch.add_argument("--system",      default=None,
                      help="Prompt de sistema inicial (default: asistente genérico)")
    p_ch.add_argument("--max-tokens",  type=int,   default=512,  dest="max_tokens",
                      help="Máximo de tokens a generar por respuesta (default: 512)")
    p_ch.add_argument("--temperature", type=float, default=0.7,
                      help="Temperatura de muestreo (0=greedy, 0.7=default, 1.0=creativo)")
    p_ch.add_argument("--top-p",       type=float, default=0.9,  dest="top_p",
                      help="Top-p (nucleus sampling, default: 0.9)")
    p_ch.add_argument("--cache-dir",   default=None,
                      help="Directorio de caché HF")
    p_ch.set_defaults(func=_cmd_chat)

    # --- Subcomando: export ---
    p_ex = subparsers.add_parser(
        "export",
        help="Fusiona un adapter LoRA con su base y exporta a safetensors o GGUF",
    )
    p_ex.add_argument("--adapter",      required=True,  help="Carpeta del adapter (generada por trainer)")
    p_ex.add_argument("--output",       required=True,  help="Carpeta de salida del modelo exportado")
    p_ex.add_argument("--model",        default=None,   help="ID del modelo base HF (opcional, se lee de meta.json)")
    p_ex.add_argument("--format",       default="safetensors", choices=["safetensors", "gguf"],
                      help="Formato de exportación (default: safetensors)")
    p_ex.add_argument("--quantization", default="q4_k_m",
                      help="Cuantización GGUF: q2_k, q4_k_m, q5_k_m, q8_0, f16 (default: q4_k_m)")
    p_ex.add_argument("--merged-dir",   default=None,   dest="merged_dir",
                      help="Ruta a modelo fusionado ya existente (evita rehacer el merge en GGUF)")
    p_ex.add_argument("--cache-dir",    default=None,   help="Directorio de caché HF")
    p_ex.set_defaults(func=_cmd_export)

    # --- Subcomando: convert-dataset (G2) ---
    p_conv = subparsers.add_parser(
        "convert-dataset",
        help="Convierte un dataset JSONL al formato de otro framework (LLaMA-Factory, Unsloth, Axolotl)",
    )
    p_conv.add_argument(
        "--input",     required=True,
        help="Archivo JSONL de entrada (formato ChatML o Alpaca)",
    )
    p_conv.add_argument(
        "--output",    required=True,
        help="Carpeta de salida (llamafactory/axolotl) o archivo .jsonl (unsloth)",
    )
    p_conv.add_argument(
        "--framework", required=True, choices=["llamafactory", "unsloth", "axolotl"],
        help="Framework destino: llamafactory | unsloth | axolotl",
    )
    p_conv.add_argument(
        "--name",      default="dataset",
        help="Nombre lógico del dataset — usado como nombre de archivo (default: dataset)",
    )
    p_conv.set_defaults(func=_cmd_convert_dataset)

    # --- Subcomando: learn (S10.3 — aprendizaje continuo) ---
    p_learn = subparsers.add_parser(
        "learn",
        help="Re-entrena un adapter con nuevos datos (aprendizaje continuo con replay buffer + rollback)",
    )
    p_learn.add_argument(
        "--adapter",    required=True,
        help="Carpeta del adapter base a actualizar (debe contener meta.json)",
    )
    p_learn.add_argument(
        "--output",     default=None,
        help="Carpeta de salida del adapter actualizado "
             "(default: <adapter>_retrained/)",
    )
    p_learn.add_argument(
        "--data",       default=None,
        help="Dataset JSONL con nuevos ejemplos. Alternativa a --auto.",
    )
    p_learn.add_argument(
        "--auto",       action="store_true",
        help="Extraer nuevos ejemplos desde el log de interacciones (--log). "
             "Incluye interacciones sin feedback negativo.",
    )
    p_learn.add_argument(
        "--log",        default="logs/interaction_log.jsonl",
        help="Ruta al log de interacciones (default: logs/interaction_log.jsonl — "
             "donde escribe el servidor). Solo se usa con --auto.",
    )
    p_learn.add_argument(
        "--base-model", default=None, dest="base_model",
        help="ID del modelo base HF (opcional, se lee de meta.json si no se indica)",
    )
    p_learn.add_argument(
        "--epochs",     type=int,   default=2,
        help="Épocas de re-entrenamiento (default: 2, menos que el entrenamiento inicial)",
    )
    p_learn.add_argument(
        "--batch-size", type=int,   default=None, dest="batch_size",
        help="Batch por GPU (auto según VRAM)",
    )
    p_learn.add_argument(
        "--lr",         type=float, default=1e-4,
        help="Learning rate (default: 1e-4 — más bajo que entrenamiento inicial)",
    )
    p_learn.add_argument(
        "--lora-r",     type=int,   default=16, dest="lora_r",
        help="Rango LoRA (default: 16, igual al adapter base)",
    )
    p_learn.add_argument(
        "--lora-alpha", type=int,   default=32, dest="lora_alpha",
        help="Alpha LoRA (default: 32)",
    )
    p_learn.add_argument(
        "--replay-buffer", type=int, default=200, dest="replay_buffer",
        help="Ejemplos históricos a mezclar en el nuevo entrenamiento "
             "(default: 200, 0 para desactivar)",
    )
    p_learn.add_argument(
        "--rollback-threshold", type=float, default=0.15, dest="rollback_threshold",
        help="Porcentaje máximo de incremento de eval_loss antes del rollback "
             "(default: 0.15 = 15%%)",
    )
    p_learn.add_argument(
        "--scheduled",  action="store_true",
        help="Modo programado: monitorea el log y dispara reentrenamiento automático "
             "cuando hay >= --min-new-examples ejemplos positivos nuevos "
             "O han pasado >= --max-days-between días desde el último ciclo. "
             "Requiere --auto y --log. Registra cada ciclo en <output>_cycle_log.jsonl.",
    )
    p_learn.add_argument(
        "--min-new-examples", type=int, default=50, dest="min_new_examples",
        help="Mínimo de ejemplos positivos nuevos para disparar en modo --scheduled "
             "(default: 50)",
    )
    p_learn.add_argument(
        "--max-days-between", type=float, default=30.0, dest="max_days_between",
        help="Días máximos entre ciclos de entrenamiento en modo --scheduled "
             "(default: 30). El ciclo se dispara aunque no haya suficientes ejemplos "
             "si ha pasado este tiempo desde el último ciclo.",
    )
    p_learn.set_defaults(func=_cmd_learn)

    # --- Subcomando: analyzer ---
    p_anl = subparsers.add_parser(
        "analyzer",
        help="Analiza un modelo HuggingFace y devuelve su configuracion LoRA optima",
    )
    p_anl.add_argument("--model",      required=True, help="ID del modelo HF. Ej: Qwen/Qwen2.5-7B-Instruct")
    p_anl.add_argument("--data",       default=None,  help="Dataset .jsonl para recomendaciones personalizadas")
    p_anl.add_argument("--gpu",        default=None,  help="GPU objetivo (auto-detecta si no se indica)")
    p_anl.add_argument("--cache-dir",  default=None,  help="Directorio de cache HF (opcional)")
    p_anl.add_argument("--lora-r",     type=int, default=16, help="Rango LoRA (default: 16)")
    p_anl.add_argument("--lora-alpha", type=int, default=32, help="Alpha LoRA (default: 32)")
    p_anl.add_argument("--json",       action="store_true",  help="Imprimir resultado como JSON")
    p_anl.set_defaults(func=_cmd_analyzer)

    # --- Subcomando: serve ---
    p_srv = subparsers.add_parser(
        "serve",
        help="Arranca el servidor REST (FastAPI/uvicorn) con un adapter o modelo",
    )
    p_srv.add_argument("--model",      required=True,
                       help="Ruta al adapter LoRA o ID HuggingFace del modelo")
    p_srv.add_argument("--host",       default="127.0.0.1",
                       help="Interfaz de red (default: 127.0.0.1 — solo local). "
                            "Usa 0.0.0.0 para exponer en red local.")
    p_srv.add_argument("--port",       type=int, default=8000,
                       help="Puerto TCP (default: 8000)")
    p_srv.add_argument("--base-model", default=None, dest="base_model",
                       help="ID del modelo base HF (solo si meta.json no lo contiene)")
    p_srv.add_argument("--api-key",    default=None, dest="api_key",
                       help="Clave Bearer para proteger la API (opcional). "
                            "Si no se indica, la API es pública.")
    p_srv.add_argument("--cache-dir",  default=None, dest="cache_dir",
                       help="Directorio de caché HF")
    p_srv.add_argument("--ui-only",    action="store_true", dest="ui_only",
                       help="Arrancar sin modelo (solo pestaña Digestor). "
                            "Útil para procesar datos sin GPU.")
    p_srv.set_defaults(func=_cmd_serve)

    # --- Subcomando: dpo ---
    p_dpo = subparsers.add_parser(
        "dpo",
        help="Entrena un adapter LoRA con DPO a partir del log de feedback humano",
    )
    p_dpo.add_argument(
        "--log", required=True,
        help="Ruta al interaction_log.jsonl generado por el servidor (POST /feedback)",
    )
    p_dpo.add_argument(
        "--output", required=True,
        help="Directorio donde se guardará el adapter DPO resultante",
    )
    p_dpo.add_argument(
        "--base-model", required=True, dest="base_model",
        help="ID del modelo base HuggingFace. Ej: Qwen/Qwen2.5-3B-Instruct",
    )
    p_dpo.add_argument(
        "--pairs-file", default=None, dest="pairs_file",
        help="Si se indica, usa este JSONL de pares en lugar de extraer del log",
    )
    p_dpo.add_argument(
        "--export-only", action="store_true", dest="export_only",
        help="Solo exporta el dataset de pares (no entrena)",
    )
    p_dpo.add_argument(
        "--pairs-out", default=None, dest="pairs_out",
        help="Ruta donde exportar los pares DPO (default: <output>/dpo_pairs.jsonl)",
    )
    p_dpo.add_argument(
        "--min-pairs", type=int, default=5, dest="min_pairs",
        help="Pares mínimos requeridos para lanzar el entrenamiento (default: 5)",
    )
    p_dpo.add_argument(
        "--epochs", type=int, default=1,
        help="Épocas de entrenamiento DPO (default: 1)",
    )
    p_dpo.add_argument(
        "--beta", type=float, default=0.1,
        help="Parámetro β DPO (default: 0.1). Mayor = más conservador",
    )
    p_dpo.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    p_dpo.add_argument(
        "--lora-r", type=int, default=16, dest="lora_r",
        help="Rango LoRA (default: 16)",
    )
    p_dpo.add_argument(
        "--cache-dir", default=None, dest="cache_dir",
        help="Directorio de caché HuggingFace (opcional)",
    )
    p_dpo.add_argument(
        "--reflection-dir", default=None, dest="reflection_dir",
        help="Directorio con la salida de 'reflect' (feedback implícito): "
             "fusiona etiquetas inferidas y pares de corrección con el "
             "feedback humano (que tiene prioridad).",
    )
    p_dpo.set_defaults(func=_cmd_dpo)

    # --- Subcomando: reflect ---
    p_ref = subparsers.add_parser(
        "reflect",
        help="Pase de reflexión: infiere feedback implícito del log con un "
             "LLM-juez (aprendizaje híbrido, estilo Hermes).",
    )
    p_ref.add_argument(
        "--log", required=True,
        help="Ruta al interaction_log.jsonl del servidor",
    )
    p_ref.add_argument(
        "--out", default="datasets/reflection", dest="out",
        help="Directorio de salida (reflection_labels.jsonl + "
             "reflection_pairs.jsonl). Default: datasets/reflection",
    )
    p_ref.add_argument(
        "--min-confidence", type=float, default=0.6, dest="min_confidence",
        help="Confianza mínima del juez para aceptar una etiqueta (0-1, "
             "default: 0.6)",
    )
    p_ref.add_argument(
        "--judge-url", default=None, dest="judge_url",
        help="Endpoint OpenAI del modelo juez (o MOTOR_JUDGE_URL; default "
             "http://localhost:8001/v1)",
    )
    p_ref.add_argument(
        "--judge-model", default=None, dest="judge_model",
        help="ID del modelo juez (o MOTOR_JUDGE_MODEL). Puede ser un modelo "
             "más fuerte que el principal para juzgar mejor.",
    )
    p_ref.set_defaults(func=_cmd_reflect)

    # --- Subcomando: cycle ---
    p_cycle = subparsers.add_parser(
        "cycle",
        help="Ciclo de mejora continua: digest → train → export → benchmark → promover",
    )
    p_cycle.add_argument(
        "--only-step",
        choices=["digest", "train", "export", "benchmark", "promote"],
        default=None, dest="only_step",
        help="Ejecutar solo un paso del ciclo",
    )
    p_cycle.add_argument(
        "--min-examples", type=int, default=50, dest="min_examples",
        help="Mínimo de ejemplos en el log para entrenar (default: 50)",
    )
    p_cycle.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Simular sin escribir nada",
    )
    p_cycle.set_defaults(func=_cmd_cycle)

    # --- Subcomando: odysseus ---
    p_odys = subparsers.add_parser(
        "odysseus",
        help="Puente de integración con Odysseus: CL bridge + watchdog + MCP tools",
    )
    p_odys.add_argument(
        "--mcp-only", action="store_true",
        help="Solo mostrar/escribir definiciones de herramientas MCP.",
    )
    p_odys.add_argument(
        "--cl-only", action="store_true",
        help="Solo ejecutar el ciclo de Continual Learning desde logs de Odysseus.",
    )
    p_odys.add_argument(
        "--watchdog-only", action="store_true",
        help="Solo iniciar el watchdog de promoción de GGUF.",
    )
    p_odys.add_argument(
        "--log", default=None, dest="odys_log",
        help="Ruta al interaction_log de Odysseus (default: logs/interaction_log.jsonl).",
    )
    p_odys.add_argument(
        "--adapter", default=None, dest="odys_adapter",
        help="Ruta al adapter base para CL (auto-detecta si no se indica).",
    )
    p_odys.add_argument(
        "--min-pairs", type=int, default=5, dest="odys_min_pairs",
        help="Pares mínimos para disparar reentrenamiento (default: 5).",
    )
    p_odys.add_argument(
        "--dry-run", action="store_true", dest="odys_dry_run",
        help="Simular sin ejecutar entrenamiento ni escritura.",
    )
    p_odys.set_defaults(func=_cmd_odysseus)

    # --- Subcomando: info ---
    p_info = subparsers.add_parser(
        "info",
        help="Inspecciona un dataset.jsonl existente",
    )
    p_info.add_argument("file", help="Ruta al archivo .jsonl a inspeccionar")
    p_info.set_defaults(func=_cmd_info)

    return parser


# ===========================================================================
# COMANDO — reflect (feedback implícito por LLM-juez, aprendizaje híbrido)
# ===========================================================================

def _cmd_reflect(args: argparse.Namespace) -> None:
    """Relee el log e infiere feedback implícito (aciertos/errores) con un
    LLM-juez, sin necesidad de clics. Complementa al feedback explícito; el
    resultado se fusiona en DPO con `dpo --reflection-dir`."""
    import os as _os
    from motor.reflection import ReflectionJudge, format_report

    if args.judge_url:
        _os.environ["MOTOR_JUDGE_URL"] = args.judge_url
    if args.judge_model:
        _os.environ["MOTOR_JUDGE_MODEL"] = args.judge_model

    judge = ReflectionJudge(args.log, min_confidence=args.min_confidence)
    try:
        result = judge.run()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(format_report(result))
    paths = judge.write(result, args.out)
    print(f"[reflect] etiquetas → {paths['labels']}")
    print(f"[reflect] pares     → {paths['pairs']}")
    print("[reflect] Úsalo en DPO con: "
          f"fabrica_loras.py dpo --log {args.log} "
          f"--reflection-dir {args.out} ...")


# ===========================================================================
# SECCIÓN 10: COMANDO — dpo (S10.4: DPO pipeline desde feedback humano)
# ===========================================================================

def _cmd_dpo(args: argparse.Namespace) -> None:
    """
    Extrae pares de preferencia del log de interacciones y entrena con DPO.

    Uso básico (extraer + entrenar):
        python fabrica_loras.py dpo \\
            --log logs/interaction_log.jsonl \\
            --output adapters/dpo_v1/ \\
            --base-model Qwen/Qwen2.5-3B-Instruct

    Solo exportar pares (sin entrenar):
        python fabrica_loras.py dpo \\
            --log logs/interaction_log.jsonl \\
            --output adapters/dpo_v1/ \\
            --base-model Qwen/Qwen2.5-3B-Instruct \\
            --export-only

    Desde un JSONL de pares ya preparado:
        python fabrica_loras.py dpo \\
            --pairs-file datasets/dpo_pairs.jsonl \\
            --output adapters/dpo_v1/ \\
            --base-model Qwen/Qwen2.5-3B-Instruct
    """
    from motor.dpo_trainer import DPOBuilder

    output_dir = args.output
    pairs_out  = args.pairs_out or str(
        __import__("pathlib").Path(output_dir) / "dpo_pairs.jsonl"
    )

    builder = DPOBuilder(
        log_path       = args.log,
        min_pairs      = args.min_pairs,
        reflection_dir = getattr(args, "reflection_dir", None),
    )

    # Mostrar estadísticas del log
    try:
        s = builder.stats()
        print(f"\n[DPO] Log analizado:")
        print(f"      Entradas con feedback : {s['rated_entries']}")
        print(f"      Positivos (👍)         : {s['positive']}")
        print(f"      Negativos (👎)         : {s['negative']}")
        print(f"      Pares disponibles      : {s['pairs_available']}")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if s["pairs_available"] == 0:
        print("\n[DPO] Sin pares de preferencia. Necesitas al menos un 👍 y un 👎 "
              "para el mismo prompt.\n"
              "  1. Usa el servidor y envía feedback positivo y negativo.\n"
              "  2. Luego vuelve a ejecutar este comando.")
        sys.exit(0)

    if s["pairs_available"] < args.min_pairs:
        print(f"\n[DPO] Solo hay {s['pairs_available']} pares "
              f"(mínimo: {args.min_pairs}). Usa --min-pairs 1 para forzar.")
        sys.exit(1)

    # Exportar pares
    try:
        out_path = builder.to_jsonl(pairs_out)
        print(f"[DPO] Pares exportados a: {out_path}")
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if args.export_only:
        print("[DPO] --export-only: entrenamiento omitido.")
        sys.exit(0)

    # Entrenar
    try:
        result_dir = builder.fit(
            output_dir            = output_dir,
            base_model_id         = args.base_model,
            dataset_path          = args.pairs_file or out_path,
            num_train_epochs      = args.epochs,
            learning_rate         = args.lr,
            beta                  = args.beta,
            lora_r                = args.lora_r,
            cache_dir             = args.cache_dir,
        )
        print(f"\n[DPO] ✅ Adapter DPO guardado en: {result_dir}")
        print(f"      Úsalo con: python fabrica_loras.py serve --model {result_dir} "
              f"--base-model {args.base_model}")
    except Exception as exc:
        print(f"[ERROR] Fallo en el entrenamiento DPO: {exc}")
        sys.exit(1)


# ===========================================================================
# SECCIÓN 11: COMANDO — cycle (ciclo de mejora continua)
# ===========================================================================

def _cmd_cycle(args: argparse.Namespace) -> None:
    """
    Ejecuta el ciclo completo de mejora continua del agente:
      1. DataDigestor  — logs/interaction_log.jsonl → dataset JSONL
      2. ContinualLearner — reentrenamiento incremental + rollback
      3. ExportManager — merge + Q4_K_M → modelos/candidate.gguf
      4. BenchmarkWorker — valida el candidato (5 tareas, umbral 80%)
      5. Promoción — escribe promotion/ready.flag

    Uso:
        python fabrica_loras.py cycle
        python fabrica_loras.py cycle --only-step digest
        python fabrica_loras.py cycle --dry-run
    """
    import motor.continual_cycle as cc
    cc.MIN_EXAMPLES = args.min_examples
    cc.run_cycle(only_step=args.only_step, dry_run=args.dry_run)


# ===========================================================================
# SECCIÓN 12: COMANDO — odysseus (puente de integración)
# ===========================================================================

def _cmd_odysseus(args: argparse.Namespace) -> None:
    """
    Puente de integración entre Motor de LoRAs y Odysseus.

    Uso:
        python fabrica_loras.py odysseus
        python fabrica_loras.py odysseus --cl-only --min-pairs 3
        python fabrica_loras.py odysseus --mcp-only > odysseus_mcp_tools.json
    """
    from motor.odysseus_bridge import (
        build_mcp_tool_definitions,
        run_cl_cycle,
        watch_promotion_dir,
        ODYSSEUS_LOG_PATH,
    )
    import json as _json
    from pathlib import Path as _Path

    all_modes = not (getattr(args, "mcp_only", False) or
                     getattr(args, "cl_only", False) or
                     getattr(args, "watchdog_only", False))

    # ── MCP Tools ──────────────────────────────────────────────────
    if all_modes or getattr(args, "mcp_only", False):
        tools = build_mcp_tool_definitions()
        print("=" * 60)
        print("  🛠️  Herramientas MCP para Odysseus")
        print("=" * 60)
        for t in tools:
            desc = t['description'][:120]
            print(f"  • {t['name']}")
            print(f"    {desc}")
        print("\n  Para registrar en Odysseus:")
        print("    Settings → MCP Servers → Add")
        print("    python fabrica_loras.py odysseus --mcp-only")
        print()
        if getattr(args, "mcp_only", False):
            print(_json.dumps(tools, indent=2, ensure_ascii=False))

    # ── CL Cycle ────────────────────────────────────────────────────
    if all_modes or getattr(args, "cl_only", False):
        log_path = (_Path(args.odys_log)
                    if getattr(args, "odys_log", None)
                    else ODYSSEUS_LOG_PATH)
        print("=" * 60)
        print("  🔄 Ciclo CL desde logs de Odysseus")
        print("=" * 60)
        result = run_cl_cycle(
            adapter_dir=getattr(args, "odys_adapter", None),
            log_path=log_path,
            min_pairs=getattr(args, "odys_min_pairs", 5),
            dry_run=getattr(args, "odys_dry_run", False),
        )
        print(f"\n  Resultado: {_json.dumps(result, indent=2, default=str)}")

    # ── Watchdog ────────────────────────────────────────────────────
    if all_modes or getattr(args, "watchdog_only", False):
        print("=" * 60)
        print("  👁️  Watchdog de promoción GGUF...")
        print("=" * 60)
        watch_promotion_dir()


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
