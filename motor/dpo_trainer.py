"""
motor.dpo_trainer
=================
Pipeline DPO (Direct Preference Optimization) a partir de feedback humano.

Flujo:
  1.  Lee interaction_log.jsonl generado por motor.server (S10.1)
  2.  Extrae entradas con feedback (rating=1 👍 o rating=-1 👎)
  3.  Forma pares (prompt, chosen, rejected) agrupando por user_msg
  4.  Exporta dataset JSONL listo para TRL DPOTrainer
  5.  Opcionalmente lanza el entrenamiento con un adapter LoRA

Formato de entrada (interaction_log.jsonl):
  {"id": "...", "timestamp": "...", "user_msg": "...", "assistant": "...",
   "feedback": 1,  "session_id": "...", "turn": 1, "model": "...", "ms": 123}

Formato de salida (dpo_pairs.jsonl):
  {"prompt": "...", "chosen": "...", "rejected": "..."}

Uso rápido:
  from motor.dpo_trainer import DPOBuilder

  builder = DPOBuilder("logs/interaction_log.jsonl")
  pairs   = builder.build_pairs()           # lista de dicts
  builder.to_jsonl("datasets/dpo_pairs.jsonl")
  builder.fit(
      output_dir     = "adapters/dpo_v1",
      base_model_id  = "Qwen/Qwen2.5-3B-Instruct",
  )

CLI:
  python fabrica_loras.py dpo --log logs/interaction_log.jsonl \\
         --output adapters/dpo_v1 --base-model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# DPOBuilder
# ---------------------------------------------------------------------------

class DPOBuilder:
    """
    Extrae pares de preferencia de un log de interacciones y entrena con DPO.

    Parámetros
    ----------
    log_path : str | Path
        Ruta al archivo interaction_log.jsonl del servidor.
    min_pairs : int
        Número mínimo de pares requeridos para poder lanzar el entrenamiento.
        Por defecto 5; reduce a 1 para pruebas.
    normalize_prompt : bool
        Si True, normaliza la clave de agrupación (minúsculas + strip).
    """

    def __init__(
        self,
        log_path: str | Path,
        min_pairs: int = 5,
        normalize_prompt: bool = True,
        reflection_dir: Optional[str | Path] = None,
    ) -> None:
        self.log_path         = Path(log_path)
        self.min_pairs        = min_pairs
        self.normalize_prompt = normalize_prompt
        # Directorio con la salida del pase de reflexión (feedback implícito):
        # reflection_labels.jsonl (ratings inferidos) + reflection_pairs.jsonl
        # (pares de corrección). Complementa al feedback humano explícito; el
        # humano SIEMPRE tiene prioridad (ver _load_reflection).
        self.reflection_dir   = Path(reflection_dir) if reflection_dir else None
        self._pairs: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Lectura y extracción
    # ------------------------------------------------------------------

    def load_entries(self) -> List[dict]:
        """
        Carga las entradas con feedback ±1 que sean texto bien formado.

        Aplica el filtro de calidad compartido (vacíos, basura tipo "[]"/"null",
        truncados, tool-calls), pero NO descarta respuestas "malas pero
        coherentes": una respuesta con 👎 es precisamente el ejemplo `rejected`
        que DPO necesita. Filtramos lo MALFORMADO, no lo DISLIKED.
        """
        from motor.log_quality import load_quality_entries, format_report

        report: dict = {}
        # dedup=False: en DPO un mismo prompt con respuestas distintas (una 👍
        # y otra 👎) es la materia prima de los pares — no deduplicar por prompt.
        # min_chars=1: en DPO el humano YA juzgó con su feedback; una respuesta
        # corta ("París") es un ejemplo legítimo. Solo filtramos lo MALFORMADO
        # (vacío/basura/truncado/tool-call), no lo corto.
        rated = load_quality_entries(
            self.log_path,
            feedback={1, -1},
            dedup=False,
            min_chars=1,
            report=report,
        )
        print(format_report(report))
        return rated

    def _load_reflection(self, explicit_ids: set) -> Tuple[List[dict], List[Dict[str, str]]]:
        """Carga la salida del pase de reflexión (feedback implícito).

        Devuelve (label_entries, direct_pairs):
          - label_entries: ratings inferidos unidos al log por `id`, con la
            misma forma que las entradas explícitas (para entrar en la
            agrupación por prompt). Se DESCARTAN los `id` que ya tienen
            feedback humano explícito — el humano manda.
          - direct_pairs: pares (prompt, chosen, rejected) de correcciones.
        """
        if not self.reflection_dir or not self.reflection_dir.exists():
            return [], []

        # Mapa id → entrada del log (para recuperar user_msg/assistant).
        by_id: Dict[str, dict] = {}
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("id"):
                        by_id[e["id"]] = e
        except FileNotFoundError:
            return [], []

        label_entries: List[dict] = []
        labels_path = self.reflection_dir / "reflection_labels.jsonl"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        v = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    vid = v.get("id")
                    if not vid or vid in explicit_ids:
                        continue  # feedback humano tiene prioridad
                    src = by_id.get(vid)
                    if not src or not src.get("user_msg") or not src.get("assistant"):
                        continue
                    label_entries.append({
                        "user_msg":  src["user_msg"],
                        "assistant": src["assistant"],
                        "feedback":  int(v.get("label", 0)),
                        "id":        vid,
                        "source":    "reflection",
                    })

        direct_pairs: List[Dict[str, str]] = []
        pairs_path = self.reflection_dir / "reflection_pairs.jsonl"
        if pairs_path.exists():
            with open(pairs_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        p = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if p.get("prompt") and p.get("chosen") and p.get("rejected"):
                        direct_pairs.append({
                            "prompt":   p["prompt"],
                            "chosen":   p["chosen"],
                            "rejected": p["rejected"],
                        })
        return label_entries, direct_pairs

    def build_pairs(self) -> List[Dict[str, str]]:
        """
        Construye pares (prompt, chosen, rejected) a partir del log.

        Agrupa entradas por user_msg y forma un par por cada combinación
        (👍, 👎) dentro del mismo grupo.

        Devuelve la lista de pares y la guarda en self._pairs.
        """
        entries = self.load_entries()

        # Feedback implícito del pase de reflexión (si se configuró): etiquetas
        # inferidas (entran en la agrupación) + pares de corrección directos.
        explicit_ids = {e["id"] for e in entries if e.get("id")}
        label_entries, direct_pairs = self._load_reflection(explicit_ids)
        entries = entries + label_entries

        # Agrupar por prompt normalizado
        positives: Dict[str, List[str]] = defaultdict(list)
        negatives: Dict[str, List[str]] = defaultdict(list)

        for entry in entries:
            key = entry["user_msg"]
            if self.normalize_prompt:
                key = key.strip().lower()
            assistant = entry.get("assistant", "")
            if entry["feedback"] == 1:
                positives[key].append(assistant)
            elif entry["feedback"] == -1:
                negatives[key].append(assistant)

        # Formar pares: chosen × rejected para cada prompt compartido
        pairs: List[Dict[str, str]] = []
        for key in positives:
            if key not in negatives:
                continue
            original_prompt = _find_original_prompt(entries, key, self.normalize_prompt)
            for chosen in positives[key]:
                for rejected in negatives[key]:
                    if chosen == rejected:
                        continue  # omitir si la respuesta es idéntica
                    pairs.append({
                        "prompt":   original_prompt,
                        "chosen":   chosen,
                        "rejected": rejected,
                    })

        # Anexar los pares de corrección directos de la reflexión (ya vienen
        # como chosen/rejected; no dependen de colisión por prompt).
        for p in direct_pairs:
            if p["chosen"] != p["rejected"]:
                pairs.append(p)

        self._pairs = pairs
        _refl = f" (+{len(label_entries)} etiquetas y {len(direct_pairs)} pares de reflexión)" if self.reflection_dir else ""
        print(
            f"[DPO] {len(entries)} entradas con feedback{_refl} → "
            f"{len(positives)} prompts con 👍, "
            f"{len(negatives)} con 👎, "
            f"{len(pairs)} pares formados."
        )
        return pairs

    # ------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------

    def to_jsonl(self, output_path: str | Path) -> Path:
        """
        Escribe los pares en formato JSONL.

        Lanza ValueError si no hay pares suficientes (< min_pairs).
        """
        if not self._pairs:
            self.build_pairs()
        if len(self._pairs) < self.min_pairs:
            raise ValueError(
                f"[DPO] Solo hay {len(self._pairs)} pares de preferencia "
                f"(mínimo requerido: {self.min_pairs}). "
                "Recoge más feedback antes de entrenar."
            )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for pair in self._pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"[DPO] Dataset guardado en {out} ({len(self._pairs)} pares)")
        return out

    # ------------------------------------------------------------------
    # Estadísticas
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Devuelve estadísticas básicas del log."""
        entries = self.load_entries()
        pos = sum(1 for e in entries if e.get("feedback") == 1)
        neg = sum(1 for e in entries if e.get("feedback") == -1)
        if not self._pairs:
            try:
                self.build_pairs()
            except Exception:
                pass
        return {
            "rated_entries":  len(entries),
            "positive":       pos,
            "negative":       neg,
            "pairs_available": len(self._pairs),
        }

    # ------------------------------------------------------------------
    # Entrenamiento DPO
    # ------------------------------------------------------------------

    def fit(
        self,
        output_dir: str | Path,
        base_model_id: str,
        *,
        dataset_path: Optional[str | Path] = None,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 8e-5,
        beta: float = 0.1,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_orpo: bool = True,
    ) -> Path:
        """
        Entrena un adapter LoRA con ORPO (por defecto) o DPO sobre los pares
        de preferencia recogidos del feedback de usuario.

        ORPO vs DPO
        -----------
        ORPO (Odds Ratio Preference Optimization) entrena con un solo modelo
        en memoria, incorporando la penalización de preferencia en la misma
        función de pérdida. DPO requiere dos modelos (activo + referencia frozen).

        En práctica esto significa:
          • ORPO en CPU  : ~6 GB RAM (modelo 3B fp32)
          • DPO en CPU   : ~12 GB RAM (modelo × 2)

        El parámetro `use_orpo=True` (por defecto) selecciona ORPO.
        Pasa `use_orpo=False` para volver al DPO clásico si es necesario.

        El formato del dataset (prompt / chosen / rejected) es idéntico
        en ambos métodos — no hay que cambiar nada upstream.

        Parámetros
        ----------
        output_dir : str | Path
            Directorio donde se guardará el adapter resultante.
        base_model_id : str
            Identificador HuggingFace del modelo base.
        use_orpo : bool
            True (default) → ORPO (1 modelo, -50% RAM).
            False          → DPO clásico (2 modelos, compatibilidad legada).
        beta : float
            En ORPO: lambda_ (peso de la penalización de preferencia, 0.1).
            En DPO:  beta     (conservadurismo respecto al modelo ref, 0.1).
        learning_rate : float
            ORPO converge mejor con LR ligeramente mayor (8e-5 vs 5e-5 de DPO).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Intentar importar ORPOTrainer; degradar a DPO si TRL < 0.8
        _trainer_cls  = None
        _config_cls   = None
        _method_label = None
        if use_orpo:
            try:
                from trl import ORPOTrainer, ORPOConfig
                _trainer_cls  = ORPOTrainer
                _config_cls   = ORPOConfig
                _method_label = "orpo"
            except ImportError:
                print("[ORPO] ORPOTrainer no disponible (TRL < 0.8) "
                      "— usando DPO como fallback")
                use_orpo = False

        if not use_orpo:
            from trl import DPOTrainer, DPOConfig
            _trainer_cls  = DPOTrainer
            _config_cls   = DPOConfig
            _method_label = "dpo"

        # Obtener pares
        if dataset_path is not None:
            pairs = _load_pairs_from_jsonl(Path(dataset_path))
        else:
            if not self._pairs:
                self.build_pairs()
            if len(self._pairs) < self.min_pairs:
                raise ValueError(
                    f"[{_method_label.upper()}] Pares insuficientes: "
                    f"{len(self._pairs)} (mínimo: {self.min_pairs})"
                )
            pairs = self._pairs

        print(f"[{_method_label.upper()}] Iniciando entrenamiento con {len(pairs)} pares")
        print(f"      Método      : {_method_label.upper()}"
              f"  ({'1 modelo en RAM' if _method_label == 'orpo' else '2 modelos en RAM'})")
        print(f"      Modelo base : {base_model_id}")
        print(f"      Salida      : {output_dir}")

        # Dispositivo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.bfloat16 if (
            device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        ) else torch.float32

        load_in_4bit = (
            bool(os.environ.get("LOAD_IN_4BIT", "false").lower() == "true")
            and device == "cuda"
        )

        # Tokenizador
        print(f"[{_method_label.upper()}] Cargando tokenizador...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            cache_dir         = cache_dir,
            trust_remote_code = True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Modelo
        print(f"[{_method_label.upper()}] Cargando modelo base...")
        model_kwargs: dict = {
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else "cpu",
        }
        if load_in_4bit:
            from motor._model_utils import apply_4bit_quantization
            apply_4bit_quantization(model_kwargs, dtype=dtype)
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = dtype

        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir

        model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

        # Adapter LoRA con rsLoRA: escala α/√r en lugar de α/r.
        # Más estable frente a cuantización GGUF Q4_K_M posterior.
        lora_cfg = LoraConfig(
            r              = lora_r,
            lora_alpha     = lora_alpha,
            lora_dropout   = lora_dropout,
            target_modules = "all-linear",
            task_type      = TaskType.CAUSAL_LM,
            use_rslora     = True,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # Dataset HuggingFace
        hf_dataset = Dataset.from_list(pairs)

        # Configuración del trainer
        # ORPO y DPO comparten casi todos los parámetros; el único que difiere
        # es el nombre del hiperparámetro de penalización (lambda_ vs beta).
        common_kwargs = dict(
            output_dir                    = str(output_dir),
            num_train_epochs              = num_train_epochs,
            per_device_train_batch_size   = per_device_train_batch_size,
            gradient_accumulation_steps   = gradient_accumulation_steps,
            learning_rate                 = learning_rate,
            max_length                    = max_length,
            remove_unused_columns         = False,
            logging_steps                 = 1,
            save_strategy                 = "epoch",
            gradient_checkpointing        = device == "cuda",
            gradient_checkpointing_kwargs = {"use_reentrant": False},
        )

        if _method_label == "orpo":
            # En ORPO, beta es el λ que pondera la pérdida de preferencia
            trainer_config = _config_cls(beta=beta, **common_kwargs)
        else:
            # DPO clásico usa beta directamente
            trainer_config = _config_cls(beta=beta, **common_kwargs)

        trainer = _trainer_cls(
            model         = model,
            args          = trainer_config,
            train_dataset = hf_dataset,
            tokenizer     = tokenizer,
        )

        t0 = time.time()
        print(f"[{_method_label.upper()}] Entrenando...")
        trainer.train()
        elapsed = round(time.time() - t0, 1)

        # Guardar adapter
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # meta.json compatible con el resto del sistema
        meta = {
            "model_id":     base_model_id,
            "base_model":   base_model_id,
            "training":     _method_label,    # "orpo" o "dpo"
            "beta":         beta,
            "pairs":        len(pairs),
            "epochs":       num_train_epochs,
            "train_time_s": elapsed,
        }
        with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[{_method_label.upper()}] ✅ Adapter guardado en {output_dir} ({elapsed}s)")
        return output_dir


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _find_original_prompt(
    entries: List[dict],
    normalized_key: str,
    normalize: bool,
) -> str:
    """Recupera el user_msg original (sin normalizar) de la primera entrada coincidente."""
    for entry in entries:
        key = entry["user_msg"].strip().lower() if normalize else entry["user_msg"]
        if key == normalized_key:
            return entry["user_msg"]
    return normalized_key  # fallback


def _load_pairs_from_jsonl(path: Path) -> List[Dict[str, str]]:
    """Carga pares DPO desde un JSONL previamente exportado."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs
