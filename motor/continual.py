"""
motor.continual
===============
ContinualLearner — Aprendizaje continuo con replay buffer y rollback automático.

Resuelve el problema del **olvido catastrófico** (catastrophic forgetting):
cuando entrenas un nuevo adapter LoRA, el modelo tiende a olvidar las tareas
aprendidas en adapters anteriores.

Dos mecanismos:
  1. Replay buffer — mezcla ejemplos de datasets pasados con el nuevo.
  2. Rollback automático — si el eval_loss sube más de un umbral, restaura
     el adapter previo desde backup.

Uso básico
----------
    from motor.continual import ContinualLearner

    cl = ContinualLearner(
        model_id         = "Qwen/Qwen2.5-3B-Instruct",
        replay_buffer_size = 200,
        rollback_threshold = 0.15,   # 15% de regresión → rollback
    )

    # Primera tarea — sin replay (no hay histórico aún)
    cl.fit(
        dataset_path = "datasets/titanic.jsonl",
        output_dir   = "adapters/titanic_v1",
        adapter_name = "titanic_v1",
    )

    # Segunda tarea — mezcla 200 ejemplos de titanic automáticamente
    cl.fit(
        dataset_path = "datasets/nueva_tarea.jsonl",
        output_dir   = "adapters/nueva_tarea_v1",
        adapter_name = "nueva_tarea_v1",
    )

    cl.history()          # Ver todos los adapters registrados
    cl.rollback("adapters/nueva_tarea_v1")  # Rollback manual si hace falta

Migrar adapters existentes
--------------------------
    cl.register_existing(
        adapter_dir  = "adapters/titanic_llm",
        dataset_path = "datasets/titanic_dataset.jsonl",
        name         = "titanic_v0",
    )
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ContinualLearner:
    """
    Gestor de aprendizaje continuo para adapters LoRA.

    Parámetros
    ----------
    model_id : str
        ID HuggingFace del modelo base. Ej: "Qwen/Qwen2.5-3B-Instruct".
    registry_path : str
        Fichero JSON central donde se registran todos los adapters.
        Por defecto "adapters/registry.json".
    replay_buffer_size : int
        Número de ejemplos de datasets pasados a mezclar con el nuevo.
        0 desactiva el replay buffer. Por defecto 200.
    rollback_threshold : float
        Máximo incremento tolerado de eval_loss frente al baseline.
        0.15 → si sube más de un 15%, se hace rollback automático.
        Por defecto 0.15.
    **trainer_kwargs
        Argumentos opcionales para LLMTrainer:
        lora_r, lora_alpha, lora_dropout, load_in_4bit,
        max_seq_length, target_modules, cache_dir.
    """

    def __init__(
        self,
        model_id: str,
        registry_path: str = "adapters/registry.json",
        replay_buffer_size: int = 200,
        rollback_threshold: float = 0.15,
        **trainer_kwargs,
    ) -> None:
        self.model_id           = model_id
        self.registry_path      = Path(registry_path)
        self.replay_buffer_size = replay_buffer_size
        self.rollback_threshold = rollback_threshold
        self._trainer_kwargs    = trainer_kwargs
        self._registry: dict    = self._load_registry()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset_path: str,
        output_dir: str,
        adapter_name: str = "",
        epochs: int = 3,
        batch_size: int = 4,
        grad_accum: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 10,
        eval_split: float = 0.1,
        logging_steps: int = 10,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Entrena un nuevo adapter con replay buffer y protección de rollback.

        Parámetros
        ----------
        dataset_path : str
            Nuevo dataset JSONL generado por DataDigestor.
        output_dir : str
            Carpeta donde se guardará el adapter.
        adapter_name : str
            Nombre descriptivo para el registro. Por defecto: basename de output_dir.
        epochs, batch_size, grad_accum, learning_rate, warmup_steps,
        eval_split, logging_steps
            Pasan directamente a LLMTrainer.fit().
        seed : int
            Semilla para el muestreo del replay buffer.

        Devuelve
        --------
        dict
            Métricas de LLMTrainer más:
            - "replay_samples_used" : int
            - "rollback_triggered"  : bool
            - "regression_pct"      : float | None
        """
        # Importación lazy — LLMTrainer requiere torch/GPU
        from motor.trainer_llm import LLMTrainer

        dataset_path = Path(dataset_path)
        output_dir   = Path(output_dir)
        adapter_name = adapter_name or output_dir.name

        print("\n" + "=" * 60)
        print(f"  ContinualLearner — adapter: {adapter_name}")
        print(f"  Adapters en registro: {len(self._registry.get('adapters', []))}")
        print(f"  Replay buffer size:   {self.replay_buffer_size}")
        print(f"  Rollback threshold:   {self.rollback_threshold:.0%}")
        print("=" * 60)

        # ── 1. Construir dataset con replay buffer ────────────────────
        merged_path, replay_count = self._build_merged_dataset(dataset_path, seed)

        backup_dir: Optional[Path] = None
        try:
            # ── 2. Backup del adapter anterior ────────────────────────
            backup_dir = self._backup_adapter(output_dir)

            # ── 3. Entrenar ───────────────────────────────────────────
            trainer = LLMTrainer(model_id=self.model_id, **self._trainer_kwargs)
            metrics = trainer.fit(
                dataset_path  = str(merged_path),
                output_dir    = str(output_dir),
                epochs        = epochs,
                batch_size    = batch_size,
                grad_accum    = grad_accum,
                learning_rate = learning_rate,
                warmup_steps  = warmup_steps,
                eval_split    = eval_split,
                logging_steps = logging_steps,
            )

            # ── 4. Detectar regresión y hacer rollback si procede ─────
            regression_pct, rollback_triggered = self._check_regression(
                new_eval_loss = metrics.get("eval_loss", 0.0),
                adapter_name  = adapter_name,
                output_dir    = output_dir,
                backup_dir    = backup_dir,
            )

            metrics["replay_samples_used"] = replay_count
            metrics["rollback_triggered"]  = rollback_triggered
            metrics["regression_pct"]      = regression_pct

            # ── 5. Registrar en el historial ──────────────────────────
            if not rollback_triggered:
                self._register_adapter(
                    name         = adapter_name,
                    output_dir   = output_dir,
                    dataset_path = dataset_path,
                    metrics      = metrics,
                    replay_count = replay_count,
                )
            else:
                print(
                    f"[ContinualLearner] ⚠ Adapter NO registrado "
                    f"(rollback activo — se mantiene versión anterior)."
                )

            return metrics

        finally:
            # Limpiar archivo temporal del dataset merged
            if merged_path != dataset_path and merged_path.exists():
                merged_path.unlink()

    def rollback(self, adapter_dir: str) -> bool:
        """
        Revierte manualmente un adapter a su versión de backup.

        Devuelve True si el rollback se completó con éxito.
        """
        adapter_dir = Path(adapter_dir)
        backup_dir  = self._backup_path(adapter_dir)
        if not backup_dir.exists():
            print(f"[ContinualLearner] No hay backup disponible para '{adapter_dir.name}'.")
            return False
        self._restore_backup(adapter_dir, backup_dir)
        print(f"[ContinualLearner] ✅ Rollback manual completado: {adapter_dir}")
        return True

    def history(self) -> None:
        """Muestra el historial de todos los adapters registrados."""
        adapters = self._registry.get("adapters", [])
        if not adapters:
            print("[ContinualLearner] Registro vacío — no hay adapters entrenados aún.")
            return

        print("\n" + "=" * 75)
        print(f"  Registro ContinualLearner — {len(adapters)} adapter(s) — {self.registry_path}")
        print("=" * 75)
        print(f"  {'#':>3}  {'Nombre':<28}  {'eval_loss':>9}  {'replay':>6}  {'tiempo':>7}  Fecha")
        print("  " + "-" * 71)
        for i, a in enumerate(adapters, 1):
            flag = " [ROLLBACK]" if a.get("rollback_triggered") else ""
            print(
                f"  {i:>3}. {a['name']:<28}  "
                f"{a.get('eval_loss', 0.0):>9.4f}  "
                f"{a.get('replay_samples_used', 0):>6}  "
                f"{a.get('elapsed_min', 0.0):>6.1f}m  "
                f"{a.get('timestamp', '?')[:16]}"
                f"{flag}"
            )
        print("=" * 75 + "\n")

    def register_existing(
        self,
        adapter_dir: str,
        dataset_path: str,
        name: str = "",
        eval_loss: float = 0.0,
    ) -> None:
        """
        Registra un adapter ya entrenado sin re-entrenarlo.

        Útil para migrar adapters previos al sistema ContinualLearner
        y que sus datos entren en el replay buffer de futuros entrenamientos.

        Parámetros
        ----------
        adapter_dir : str
            Carpeta del adapter (debe contener adapter_config.json o meta.json).
        dataset_path : str
            Dataset con el que fue entrenado originalmente.
        name : str
            Nombre descriptivo. Por defecto: basename de adapter_dir.
        eval_loss : float
            eval_loss conocido. Si existe meta.json se lee de ahí automáticamente.
        """
        adapter_dir  = Path(adapter_dir)
        dataset_path = Path(dataset_path)
        name         = name or adapter_dir.name

        # Intentar leer meta.json generado por LLMTrainer
        meta_path = adapter_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            eval_loss = meta.get("eval_loss", eval_loss)

        self._register_adapter(
            name         = name,
            output_dir   = adapter_dir,
            dataset_path = dataset_path,
            metrics      = {"eval_loss": eval_loss, "train_loss": 0.0, "elapsed_min": 0.0, "train_samples": 0},
            replay_count = 0,
        )
        print(f"[ContinualLearner] Adapter existente '{name}' migrado al registro.")

    def get_registry(self) -> dict:
        """Devuelve el registro completo como dict."""
        return dict(self._registry)

    def stack_adapters(
        self,
        base_adapter_dir: str,
        personal_adapter_dir: str,
        output_dir: str,
        base_weight: float = 0.7,
        personal_weight: float = 0.3,
        combination_type: str = "linear",
    ) -> str:
        """
        Combina un adapter base (genérico) y un adapter personal (específico
        del usuario) en uno solo, usando PEFT ``add_weighted_adapter()``.

        El resultado es un adapter único que puede cargarse directamente con
        ``PeftModel.from_pretrained()`` o exportarse a GGUF.

        Parámetros
        ----------
        base_adapter_dir : str
            Carpeta del adapter base (ej. ``adapters/domestic_v2/``).
        personal_adapter_dir : str
            Carpeta del adapter personal (ej. ``adapters/personal_felipe/``).
        output_dir : str
            Carpeta de destino del adapter combinado.
        base_weight : float
            Peso del adapter base en la combinación. Por defecto 0.7.
        personal_weight : float
            Peso del adapter personal en la combinación. Por defecto 0.3.
        combination_type : str
            Método de PEFT: ``"linear"`` (suma ponderada) o ``"svd"``
            (descomposición SVD, más preciso pero más lento). Por defecto ``"linear"``.

        Devuelve
        --------
        str
            Ruta al directorio del adapter combinado (``output_dir``).

        Ejemplo
        -------
        >>> cl = ContinualLearner(model_id="Qwen/Qwen2.5-3B-Instruct")
        >>> cl.stack_adapters(
        ...     base_adapter_dir     = "adapters/domestic_v2/",
        ...     personal_adapter_dir = "adapters/personal_felipe/",
        ...     output_dir           = "adapters/stacked_felipe/",
        ...     base_weight          = 0.7,
        ...     personal_weight      = 0.3,
        ... )
        'adapters/stacked_felipe/'
        """
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        base_adapter_dir     = Path(base_adapter_dir)
        personal_adapter_dir = Path(personal_adapter_dir)
        output_dir           = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"  ContinualLearner — stack_adapters")
        print(f"  base     : {base_adapter_dir.name}  (w={base_weight})")
        print(f"  personal : {personal_adapter_dir.name}  (w={personal_weight})")
        print(f"  método   : {combination_type}")
        print("=" * 60)

        # Determinar dtype — usar float16 para compatibilidad amplia
        dtype = torch.float16

        print(f"[ContinualLearner] Cargando modelo base: {self.model_id} …")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map={"": "cpu"},   # stacking en CPU para no necesitar GPU
        )

        # Cargar adapter base con nombre "base"
        print(f"[ContinualLearner] Cargando adapter base: {base_adapter_dir.name} …")
        model = PeftModel.from_pretrained(
            model,
            str(base_adapter_dir),
            adapter_name="base",
        )

        # Cargar adapter personal con nombre "personal"
        print(f"[ContinualLearner] Cargando adapter personal: {personal_adapter_dir.name} …")
        model.load_adapter(
            str(personal_adapter_dir),
            adapter_name="personal",
        )

        # Combinar — PEFT add_weighted_adapter
        print(f"[ContinualLearner] Combinando con {combination_type} ({base_weight}/{personal_weight}) …")
        model.add_weighted_adapter(
            adapters          = ["base", "personal"],
            weights           = [base_weight, personal_weight],
            adapter_name      = "stacked",
            combination_type  = combination_type,
        )
        model.set_adapter("stacked")

        # Guardar adapter combinado
        model.save_pretrained(str(output_dir))

        # Copiar tokenizer desde el adapter base (si lo tiene)
        tokenizer_src = base_adapter_dir / "tokenizer"
        tokenizer_dst = output_dir / "tokenizer"
        if tokenizer_src.exists() and not tokenizer_dst.exists():
            shutil.copytree(str(tokenizer_src), str(tokenizer_dst))

        # Guardar meta.json con información de la combinación
        meta = {
            "base_adapter":        str(base_adapter_dir),
            "personal_adapter":    str(personal_adapter_dir),
            "base_weight":         base_weight,
            "personal_weight":     personal_weight,
            "combination_type":    combination_type,
            "model_id":            self.model_id,
            "stacked":             True,
            "created":             datetime.now().isoformat(),
        }
        with open(output_dir / "meta.json", "w", encoding="utf-8") as _f:
            json.dump(meta, _f, indent=2, ensure_ascii=False)

        print(f"[ContinualLearner] ✅ Adapter combinado guardado en: {output_dir}")
        return str(output_dir)

    # ------------------------------------------------------------------
    # Privados — replay buffer
    # ------------------------------------------------------------------

    def _build_merged_dataset(
        self,
        new_dataset: Path,
        seed: int,
    ) -> Tuple[Path, int]:
        """
        Mezcla el nuevo dataset con muestras del replay buffer.

        Devuelve (ruta_del_dataset_merged, n_ejemplos_de_replay).
        Si no hay histórico, devuelve el dataset original sin crear archivo.
        """
        replay_examples = self._sample_replay_buffer(seed)
        if not replay_examples:
            print("[ContinualLearner] Sin replay buffer (primer adapter o desactivado).")
            return new_dataset, 0

        # Leer nuevo dataset
        new_lines = [
            line for line in new_dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # Serializar replay + mezclar
        replay_lines = [json.dumps(ex, ensure_ascii=False) for ex in replay_examples]
        all_lines    = new_lines + replay_lines
        random.Random(seed).shuffle(all_lines)

        # Escribir en archivo temporal
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
            encoding="utf-8", prefix="cl_merged_",
        )
        tmp.write("\n".join(all_lines))
        tmp.close()

        print(
            f"[ContinualLearner] Replay buffer: {len(replay_examples)} ejemplos históricos "
            f"mezclados con {len(new_lines)} nuevos → {len(all_lines)} total."
        )
        return Path(tmp.name), len(replay_examples)

    def _sample_replay_buffer(self, seed: int) -> List[dict]:
        """
        Toma una muestra aleatoria de todos los datasets históricos registrados.

        Distribuye el presupuesto (replay_buffer_size) equitativamente entre
        todos los adapters registrados cuyo dataset sigue existiendo en disco.
        """
        if self.replay_buffer_size == 0:
            return []

        past_datasets: List[str] = [
            a["dataset_path"]
            for a in self._registry.get("adapters", [])
            if Path(a.get("dataset_path", "")).exists()
        ]
        if not past_datasets:
            return []

        rng        = random.Random(seed)
        per_source = max(1, self.replay_buffer_size // len(past_datasets))
        samples: List[dict] = []

        for ds_path in past_datasets:
            try:
                lines = [
                    line for line in Path(ds_path).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                chosen = rng.sample(lines, min(per_source, len(lines)))
                samples.extend(json.loads(line) for line in chosen)
            except Exception as exc:
                print(f"[ContinualLearner] ⚠ No se pudo leer dataset pasado {ds_path}: {exc}")

        # Truncar al tamaño máximo por si acaso
        rng.shuffle(samples)
        return samples[: self.replay_buffer_size]

    # ------------------------------------------------------------------
    # Privados — backup / rollback
    # ------------------------------------------------------------------

    def _backup_adapter(self, adapter_dir: Path) -> Optional[Path]:
        """Copia el adapter actual a <adapter_dir>_backup antes de sobreescribirlo."""
        if not adapter_dir.exists():
            return None
        backup_dir = self._backup_path(adapter_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(adapter_dir, backup_dir)
        print(f"[ContinualLearner] Backup guardado: {backup_dir.name}")
        return backup_dir

    def _restore_backup(self, adapter_dir: Path, backup_dir: Path) -> None:
        """Restaura el backup al directorio original y elimina el backup."""
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        shutil.copytree(backup_dir, adapter_dir)
        shutil.rmtree(backup_dir)
        print(f"[ContinualLearner] Adapter restaurado desde backup.")

    @staticmethod
    def _backup_path(adapter_dir: Path) -> Path:
        return adapter_dir.parent / (adapter_dir.name + "_backup")

    # ------------------------------------------------------------------
    # Privados — detección de regresión
    # ------------------------------------------------------------------

    def _check_regression(
        self,
        new_eval_loss: float,
        adapter_name: str,
        output_dir: Path,
        backup_dir: Optional[Path],
    ) -> Tuple[Optional[float], bool]:
        """
        Compara new_eval_loss con el baseline del registro.

        Hace rollback automático si la regresión supera rollback_threshold.
        Devuelve (regression_pct, rollback_triggered).

        Nota sobre el eval_loss comparado
        -----------------------------------
        El eval_loss medido durante el entrenamiento incluye el split de
        evaluación del dataset MERGED (nuevo + replay). Es un proxy válido
        del rendimiento general, no una evaluación aislada por tarea.
        Para evaluación por tarea, pasa los datasets individualmente a
        LLMTrainer y compara externamente.
        """
        baseline_loss = self._get_baseline_loss(adapter_name)
        if baseline_loss is None or baseline_loss == 0.0:
            print("[ContinualLearner] Sin baseline previo — regresión no comprobada.")
            return None, False

        if new_eval_loss == 0.0:
            print("[ContinualLearner] eval_loss=0.0 — regresión no comprobada.")
            return None, False

        regression_pct = (new_eval_loss - baseline_loss) / baseline_loss

        color = "✅" if regression_pct <= self.rollback_threshold else "⚠"
        print(
            f"[ContinualLearner] {color}  "
            f"eval_loss: {baseline_loss:.4f} → {new_eval_loss:.4f}  "
            f"({regression_pct:+.1%})"
        )

        if regression_pct > self.rollback_threshold:
            print(
                f"[ContinualLearner] REGRESIÓN DETECTADA "
                f"({regression_pct:.1%} > umbral {self.rollback_threshold:.0%})"
            )
            if backup_dir and backup_dir.exists():
                self._restore_backup(output_dir, backup_dir)
                print("[ContinualLearner] ✅ Rollback automático ejecutado.")
                return regression_pct, True
            else:
                print("[ContinualLearner] ⚠ Sin backup disponible — rollback omitido.")

        return regression_pct, False

    def _get_baseline_loss(self, adapter_name: str) -> Optional[float]:
        """
        Devuelve el eval_loss de referencia para este adapter.

        Estrategia:
        1. Si hay versiones previas con el mismo nombre → usa la más reciente.
        2. Si no → usa el último adapter registrado (cualquier nombre).
        3. Si el registro está vacío → devuelve None.
        """
        adapters = self._registry.get("adapters", [])
        # Buscar por nombre exacto (más reciente primero)
        matches = [a for a in reversed(adapters) if a.get("name") == adapter_name]
        if matches:
            return matches[0].get("eval_loss")
        # Referencia global: último adapter registrado
        if adapters:
            return adapters[-1].get("eval_loss")
        return None

    def from_user_profile(
        self,
        profile: Dict[str, Any],
        base_adapter_dir: str,
        output_dir: str,
        n_examples: int = 50,
        epochs: int = 2,
        learning_rate: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Genera un dataset personalizado a partir de un perfil de usuario y
        entrena un adapter LoRA personal encima del adapter base doméstico.

        El perfil debe contener las preferencias del usuario: nombre, idioma,
        carpetas, contactos, reglas de spam, etc.

        Estructura del perfil
        ---------------------
        {
            "name": "Felipe",
            "language": "es",
            "folders": {
                "documents": "~/Documentos",
                "downloads": "~/Descargas",
                "projects": "~/Proyectos",
                "music": "~/Música",
            },
            "contacts": {
                "boss": "ana.garcia@empresa.com",
                "bank": "notificaciones@banco.es",
            },
            "rules": [
                {"type": "spam", "sender": "ofertas@promo.net"},
                {"type": "archive", "sender": "newsletter@tech.com"},
            ],
        }

        Parámetros
        ----------
        profile : dict
            Perfil del usuario con sus preferencias.
        base_adapter_dir : str
            Carpeta del adapter base doméstico (ej. adapters/domestic_v2/).
        output_dir : str
            Carpeta de destino del adapter personal.
        n_examples : int
            Número de ejemplos a generar (default 50).
        epochs : int
            Épocas de entrenamiento (default 2, dataset pequeño).
        learning_rate : float
            Learning rate (default 1e-4).

        Devuelve
        --------
        dict
            Métricas de entrenamiento + ruta del adapter personal.
        """
        import tempfile

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name     = profile.get("name", "usuario")
        language = profile.get("language", "es")
        folders  = profile.get("folders", {})
        contacts = profile.get("contacts", {})
        rules    = profile.get("rules", [])

        print("\n" + "=" * 60)
        print(f"  ContinualLearner — from_user_profile")
        print(f"  Usuario : {name}")
        print(f"  Idioma  : {language}")
        print(f"  Ejemplos: {n_examples}")
        print("=" * 60)

        # ── 1. Generar dataset personalizado ─────────────────────────
        examples = self._generate_profile_examples(
            name=name,
            language=language,
            folders=folders,
            contacts=contacts,
            rules=rules,
            n=n_examples,
        )

        # Guardar dataset temporal
        tmp_ds = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
            encoding="utf-8", prefix=f"profile_{name}_",
        )
        for ex in examples:
            tmp_ds.write(json.dumps(ex, ensure_ascii=False) + "\n")
        tmp_ds.close()
        dataset_path = Path(tmp_ds.name)

        print(f"[ContinualLearner] Dataset personal generado: {len(examples)} ejemplos → {dataset_path}")

        # ── 2. Entrenar adapter personal ─────────────────────────────
        try:
            metrics = self.fit(
                dataset_path  = str(dataset_path),
                output_dir    = str(output_dir),
                adapter_name  = f"personal_{name}",
                epochs        = epochs,
                batch_size    = 2,
                grad_accum    = 4,
                learning_rate = learning_rate,
                warmup_steps  = 5,
                eval_split    = 0.1,
                logging_steps = 5,
            )
        finally:
            # Limpiar dataset temporal
            if dataset_path.exists():
                dataset_path.unlink()

        # ── 3. Guardar perfil junto al adapter ────────────────────────
        profile_path = output_dir / "user_profile.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        # ── 4. Stacking opcional: combinar con base ──────────────────
        stacked_dir = output_dir.parent / f"{output_dir.name}_stacked"
        try:
            combined_path = self.stack_adapters(
                base_adapter_dir     = str(base_adapter_dir),
                personal_adapter_dir = str(output_dir),
                output_dir           = str(stacked_dir),
                base_weight          = 0.7,
                personal_weight      = 0.3,
            )
            metrics["stacked_adapter"] = combined_path
        except Exception as exc:
            print(f"[ContinualLearner] ⚠ Stacking omitido (requiere GPU/torch): {exc}")
            metrics["stacked_adapter"] = None

        return metrics

    # ------------------------------------------------------------------
    # Privados — generación de ejemplos desde perfil
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_profile_examples(
        name: str,
        language: str,
        folders: Dict[str, str],
        contacts: Dict[str, str],
        rules: List[dict],
        n: int,
    ) -> List[dict]:
        """
        Genera ejemplos de entrenamiento personalizados para el usuario.

        Cada ejemplo sigue el formato ChatML estándar con mensajes.
        Prioriza herramientas domésticas: file_organize, email_filter,
        note_save, search_files, calendar_get.
        """
        rng = random.Random(hash(name) + n)

        # Pool de plantillas por tipo de herramienta
        templates_file: list = []
        templates_email: list = []
        templates_note: list = []
        templates_search: list = []

        docs  = folders.get("documents", "~/Documentos")
        down  = folders.get("downloads", "~/Descargas")
        proj  = folders.get("projects", "~/Proyectos")
        music = folders.get("music", "~/Música")

        # --- file_organize ---
        if docs:
            templates_file.append({
                "user": f"Mueve los PDFs de {down} a {docs}",
                "thought": f"El usuario quiere organizar PDFs de {down} a {docs}.",
                "action": "file_organize",
                "args": {"files": [f"{down}/*.pdf"], "dest": f"{docs}/PDFs/", "dry_run": True},
                "result": f"[dry_run] file_organize: 3 PDFs listos para mover a {docs}/PDFs/.",
                "final": f"Encontré 3 archivos PDF en {down}. Usa dry_run=False para moverlos a {docs}/PDFs/.",
            })
            templates_file.append({
                "user": f"Organiza las hojas de cálculo de {docs} en su propia carpeta",
                "thought": f"Busco archivos Excel/CSV en {docs}.",
                "action": "file_organize",
                "args": {"files": [f"{docs}/*.xlsx", f"{docs}/*.csv"], "dest": f"{docs}/Hojas de cálculo/", "dry_run": True},
                "result": f"[dry_run] file_organize: 2 archivos listos para mover a {docs}/Hojas de cálculo/.",
                "final": f"Encontré 2 hojas de cálculo en {docs}. Ejecuta con dry_run=False para organizarlas.",
            })

        # --- email_filter ---
        for role, email in contacts.items():
            templates_email.append({
                "user": f"¿Tengo correos nuevos de {role}?",
                "thought": f"Busco correos de {email} en INBOX.",
                "action": "email_filter",
                "args": {"server": "imap.gmail.com", "user": f"{name.lower()}@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "sender": email, "mode": "list", "limit": 5},
                "result": f"email_filter: 2 mensajes de {email} en INBOX.",
                "final": f"Tienes 2 correos recientes de {role} ({email}).",
            })

        for rule in rules:
            if rule.get("type") == "spam":
                sender = rule.get("sender", "spam@example.com")
                templates_email.append({
                    "user": f"Marca como spam los correos de {sender}",
                    "thought": f"El usuario quiere marcar {sender} como spam.",
                    "action": "email_filter",
                    "args": {"server": "imap.gmail.com", "user": f"{name.lower()}@gmail.com", "password": "app-pass-1234", "folder": "INBOX", "sender": sender, "mode": "mark_spam", "limit": 50, "dry_run": True},
                    "result": f"[dry_run] email_filter: 5 mensajes de {sender} identificados para marcar como spam.",
                    "final": f"Encontré 5 correos de {sender}. Pasa dry_run=False para marcarlos como spam.",
                })

        # --- note_save ---
        templates_note.append({
            "user": f"Guarda una nota con las ideas para el proyecto {name}",
            "thought": "El usuario quiere guardar una nota.",
            "action": "note_save",
            "args": {"title": f"Ideas proyecto {name}", "body": f"Lista de ideas para el proyecto personal de {name}.\n- Idea 1: optimizar flujo de trabajo\n- Idea 2: automatizar tareas repetitivas", "notebook": f"Proyectos de {name}"},
            "result": f"Nota guardada en ~/Notas/Proyectos de {name}/ideas_proyecto_{name.lower()}.txt",
            "final": f"He guardado la nota 'Ideas proyecto {name}' en tu carpeta de Notas.",
        })

        # --- search_files ---
        templates_search.append({
            "user": f"Busca archivos que mencionen '{name}' en {docs}",
            "thought": f"Busco referencias a {name} en documentos.",
            "action": "search_files",
            "args": {"query": name, "path": docs, "extensions": [".txt", ".md", ".pdf", ".docx"]},
            "result": f"search_files: 1 coincidencia en {docs}/notas_personales.txt.",
            "final": f"Encontré 1 archivo que menciona '{name}' en {docs}.",
        })
        templates_search.append({
            "user": f"Encuentra todos los .py en {proj}",
            "thought": f"Busco archivos Python en {proj}.",
            "action": "search_files",
            "args": {"query": "", "path": proj, "extensions": [".py"]},
            "result": f"search_files: 12 archivos .py encontrados en {proj}.",
            "final": f"Encontré 12 archivos Python en {proj}.",
        })

        # Construir ejemplos a partir de plantillas disponibles
        all_templates = templates_file + templates_email + templates_note + templates_search
        rng.shuffle(all_templates)

        examples: list = []
        for tmpl in all_templates[:n]:
            msgs = [
                {"role": "user", "content": tmpl["user"]},
                {"role": "assistant", "content": (
                    f"Thought: {tmpl['thought']}\n"
                    f"Action: {tmpl['action']}\n"
                    f"Action Input: {json.dumps(tmpl['args'], ensure_ascii=False)}\n"
                    f"Observation: {tmpl['result']}\n"
                    f"Thought: Ya tengo el resultado.\n"
                    f"Final Answer: {tmpl['final']}"
                )},
            ]
            examples.append({"messages": msgs})

        # Si no hay suficientes plantillas, duplicar con variación
        while len(examples) < n:
            base = rng.choice(all_templates)
            msgs = [
                {"role": "user", "content": base["user"] + " por favor"},
                {"role": "assistant", "content": (
                    f"Thought: {base['thought']}\n"
                    f"Action: {base['action']}\n"
                    f"Action Input: {json.dumps(base['args'], ensure_ascii=False)}\n"
                    f"Observation: {base['result']}\n"
                    f"Thought: Proceso completado.\n"
                    f"Final Answer: {base['final']}"
                )},
            ]
            examples.append({"messages": msgs})

        return examples


    # ------------------------------------------------------------------
    # Privados — registro
    # ------------------------------------------------------------------

    def _register_adapter(
        self,
        name: str,
        output_dir: Path,
        dataset_path: Path,
        metrics: dict,
        replay_count: int,
    ) -> None:
        """Añade entrada al registro y guarda registry.json."""
        entry: Dict[str, Any] = {
            "name":                 name,
            "adapter_dir":          str(output_dir),
            "dataset_path":         str(dataset_path),
            "model_id":             self.model_id,
            "eval_loss":            metrics.get("eval_loss", 0.0),
            "train_loss":           metrics.get("train_loss", 0.0),
            "elapsed_min":          metrics.get("elapsed_min", 0.0),
            "train_samples":        metrics.get("train_samples", 0),
            "replay_samples_used":  replay_count,
            "rollback_triggered":   metrics.get("rollback_triggered", False),
            "timestamp":            datetime.now().isoformat(timespec="seconds"),
        }
        self._registry.setdefault("adapters", []).append(entry)
        self._save_registry()
        print(f"[ContinualLearner] ✅ Adapter '{name}' guardado en registro.")

    def _load_registry(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path, encoding="utf-8") as f:
                return json.load(f)
        return {"version": 1, "adapters": []}

    def _save_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)
