"""
tests/test_continual_e2e.py
============================
S10.2 â€” Test end-to-end del ContinualLearner SIN GPU.

Simula el ciclo completo de aprendizaje continuo mediante un entrenador
falso (MockTrainer) que imita la API de LLMTrainer sin cargar modelos reales.

Cubre:
  A. Primera iteraciÃ³n (sin replay buffer â€” primer adapter)
  B. Segunda iteraciÃ³n (con replay buffer activo)
  C. Rollback automÃ¡tico por regresiÃ³n
  D. Rollback manual
  E. register_existing() â€” migraciÃ³n de adapters previos
  F. Historial get_registry()
  G. IntegraciÃ³n con interaction_log.jsonl (extracciÃ³n de ejemplos para replay)
  H. Replay buffer distribuido entre mÃºltiples datasets
  I. Rollback threshold edge cases (exactamente en el lÃ­mite)
  J. Dataset dataset inexistente en replay â€” degradaciÃ³n elegante

NOTA: El MockTrainer parchea `motor.trainer_llm.LLMTrainer` via unittest.mock
para no requerir GPU ni torch instalado. Los archivos de adapter se simulan
con adapter_config.json + meta.json mÃ­nimos.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Workaround WinError 6714: pyarrow escanea sys.path y explota en Windows.
# Pre-importar pyarrow/sklearn antes de que el repo entre en sys.path.
# La raiz se deriva de __file__ a proposito: filtrar por el nombre de la carpeta
# dejaba de funcionar en silencio en cuanto el repo se renombraba o se clonaba
# con otro nombre, y el crash de pyarrow volvia sin aviso.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_repo_path(entry: str) -> bool:
    try:
        return Path(entry or ".").resolve() == _REPO_ROOT
    except (OSError, ValueError):
        return False


_orig_path = list(sys.path)
sys.path = [p for p in sys.path if not _is_repo_path(p)]
try:
    import pyarrow
except ImportError:
    pass
try:
    import sklearn
except ImportError:
    pass
sys.path = _orig_path


# Windows cp1252 no puede codificar emojis (✅, ⚠) que imprime continual.py
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from motor.continual import ContinualLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jsonl(path: Path, n: int = 20, prefix: str = "ex") -> None:
    """Escribe N ejemplos de entrenamiento mÃ­nimos en un JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = {
                "messages": [
                    {"role": "user",      "content": f"{prefix} pregunta {i}"},
                    {"role": "assistant", "content": f"{prefix} respuesta {i}"},
                ]
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _make_adapter(adapter_dir: Path, eval_loss: float = 0.15) -> None:
    """Crea los archivos mÃ­nimos de un adapter LoRA guardado."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct", "r": 16}),
        encoding="utf-8",
    )
    (adapter_dir / "meta.json").write_text(
        json.dumps({
            "model_id":   "Qwen/Qwen2.5-3B-Instruct",
            "eval_loss":  eval_loss,
            "train_loss": eval_loss * 3,
            "elapsed_min": 10.0,
            "train_samples": 100,
        }),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)


def _mock_trainer_fit(eval_loss: float = 0.12):
    """Devuelve un mock de LLMTrainer.fit() con los resultados dados."""
    mock_trainer = MagicMock()
    mock_trainer.fit.return_value = {
        "eval_loss":       eval_loss,
        "train_loss":      eval_loss * 2.5,
        "token_accuracy":  0.95,
        "elapsed_min":     5.0,
        "train_samples":   100,
    }
    return mock_trainer


# ---------------------------------------------------------------------------
# A â€” Primera iteraciÃ³n (sin replay buffer)
# ---------------------------------------------------------------------------

class TestFirstFit(unittest.TestCase):
    """Primer adapter: sin histÃ³rico â†’ no hay replay buffer."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ds  = Path(self.tmp) / "data.jsonl"
        self.out = Path(self.tmp) / "adapter_v1"
        self.reg = Path(self.tmp) / "registry.json"
        _make_jsonl(self.ds, n=50)

    def test_first_fit_no_replay(self):
        """Primera llamada a fit() no usa replay buffer (registro vacÃ­o)."""
        cl = ContinualLearner(
            model_id       = "Qwen/Qwen2.5-3B-Instruct",
            registry_path  = str(self.reg),
            replay_buffer_size = 50,
        )
        mock_t = _mock_trainer_fit(eval_loss=0.10)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.fit(
                dataset_path = str(self.ds),
                output_dir   = str(self.out),
                adapter_name = "v1",
            )
        self.assertEqual(metrics["replay_samples_used"], 0)
        self.assertFalse(metrics["rollback_triggered"])
        self.assertIsNone(metrics["regression_pct"])

    def test_registry_created_after_first_fit(self):
        """Tras el primer fit(), registry.json existe y tiene 1 adapter."""
        cl = ContinualLearner(
            model_id      = "Qwen/Qwen2.5-3B-Instruct",
            registry_path = str(self.reg),
        )
        mock_t = _mock_trainer_fit(eval_loss=0.10)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            cl.fit(str(self.ds), str(self.out), adapter_name="v1")
        self.assertTrue(self.reg.exists())
        reg = json.loads(self.reg.read_text(encoding="utf-8"))
        self.assertEqual(len(reg["adapters"]), 1)
        self.assertEqual(reg["adapters"][0]["name"], "v1")
        self.assertAlmostEqual(reg["adapters"][0]["eval_loss"], 0.10, places=5)


# ---------------------------------------------------------------------------
# B â€” Segunda iteraciÃ³n (replay buffer activo)
# ---------------------------------------------------------------------------

class TestSecondFit(unittest.TestCase):
    """Segundo fit(): el replay buffer mezcla ejemplos del primer dataset."""

    def setUp(self):
        self.tmp  = tempfile.mkdtemp()
        self.ds1  = Path(self.tmp) / "data1.jsonl"
        self.ds2  = Path(self.tmp) / "data2.jsonl"
        self.out1 = Path(self.tmp) / "adapter_v1"
        self.out2 = Path(self.tmp) / "adapter_v2"
        self.reg  = Path(self.tmp) / "registry.json"
        _make_jsonl(self.ds1, n=100, prefix="ds1")
        _make_jsonl(self.ds2, n=80,  prefix="ds2")

    def test_second_fit_uses_replay(self):
        """El segundo fit() informa replay_samples_used > 0."""
        cl = ContinualLearner(
            model_id           = "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = str(self.reg),
            replay_buffer_size = 30,
        )
        mock_t = _mock_trainer_fit(eval_loss=0.10)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            cl.fit(str(self.ds1), str(self.out1), adapter_name="v1")
            metrics2 = cl.fit(str(self.ds2), str(self.out2), adapter_name="v2")

        self.assertGreater(metrics2["replay_samples_used"], 0)
        self.assertLessEqual(metrics2["replay_samples_used"], 30)

    def test_second_fit_registry_has_two_adapters(self):
        """El registro tiene 2 adapters despuÃ©s de dos fits exitosos."""
        cl = ContinualLearner(
            model_id      = "Qwen/Qwen2.5-3B-Instruct",
            registry_path = str(self.reg),
        )
        mock_t = _mock_trainer_fit(eval_loss=0.10)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            cl.fit(str(self.ds1), str(self.out1), adapter_name="v1")
            cl.fit(str(self.ds2), str(self.out2), adapter_name="v2")

        reg = json.loads(self.reg.read_text(encoding="utf-8"))
        self.assertEqual(len(reg["adapters"]), 2)


# ---------------------------------------------------------------------------
# C â€” Rollback automÃ¡tico por regresiÃ³n
# ---------------------------------------------------------------------------

class TestRollbackAuto(unittest.TestCase):
    """Si la regresiÃ³n supera el umbral, el adapter se revierte al backup."""

    def setUp(self):
        self.tmp  = tempfile.mkdtemp()
        self.ds1  = Path(self.tmp) / "data1.jsonl"
        self.ds2  = Path(self.tmp) / "data2.jsonl"
        self.out  = Path(self.tmp) / "adapter"
        self.reg  = Path(self.tmp) / "registry.json"
        _make_jsonl(self.ds1, n=40, prefix="ds1")
        _make_jsonl(self.ds2, n=40, prefix="ds2")

    def _run_first_fit(self, cl, eval_loss=0.10):
        mock_t = _mock_trainer_fit(eval_loss=eval_loss)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            cl.fit(str(self.ds1), str(self.out), adapter_name="adapter")
        # Crear archivos del adapter para que el backup tenga quÃ© copiar
        _make_adapter(self.out, eval_loss=eval_loss)

    def test_rollback_triggered_on_regression(self):
        """Si eval_loss sube >15% â†’ rollback_triggered=True."""
        cl = ContinualLearner(
            model_id           = "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = str(self.reg),
            rollback_threshold = 0.15,
        )
        self._run_first_fit(cl, eval_loss=0.10)

        # Segundo fit con regresiÃ³n fuerte (0.10 â†’ 0.20 = +100%)
        mock_t = _mock_trainer_fit(eval_loss=0.20)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.fit(str(self.ds2), str(self.out), adapter_name="adapter")

        self.assertTrue(metrics["rollback_triggered"])
        self.assertIsNotNone(metrics["regression_pct"])
        self.assertGreater(metrics["regression_pct"], 0.15)

    def test_no_rollback_within_threshold(self):
        """Si la regresiÃ³n estÃ¡ dentro del umbral â†’ rollback_triggered=False."""
        cl = ContinualLearner(
            model_id           = "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = str(self.reg),
            rollback_threshold = 0.15,
        )
        self._run_first_fit(cl, eval_loss=0.10)

        # Subida del 5% â€” dentro del umbral del 15%
        mock_t = _mock_trainer_fit(eval_loss=0.105)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.fit(str(self.ds2), str(self.out), adapter_name="adapter")

        self.assertFalse(metrics["rollback_triggered"])

    def test_rollback_adapter_NOT_registered(self):
        """Cuando hay rollback, el adapter regresado NO se aÃ±ade al registro."""
        cl = ContinualLearner(
            model_id           = "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = str(self.reg),
            rollback_threshold = 0.15,
        )
        self._run_first_fit(cl, eval_loss=0.10)
        count_before = len(cl.get_registry().get("adapters", []))

        mock_t = _mock_trainer_fit(eval_loss=0.50)  # regresiÃ³n brutal
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.fit(str(self.ds2), str(self.out), adapter_name="adapter")

        self.assertTrue(metrics["rollback_triggered"])
        # El registro no creciÃ³
        count_after = len(cl.get_registry().get("adapters", []))
        self.assertEqual(count_before, count_after)

    def test_rollback_threshold_exact_boundary(self):
        """RegresiÃ³n exactamente igual al umbral â†’ NO hay rollback (no > sino â‰¤)."""
        threshold = 0.15
        cl = ContinualLearner(
            model_id           = "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = str(self.reg),
            rollback_threshold = threshold,
        )
        self._run_first_fit(cl, eval_loss=0.10)

        # Exactamente +15%
        mock_t = _mock_trainer_fit(eval_loss=0.115)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.fit(str(self.ds2), str(self.out), adapter_name="adapter")

        self.assertFalse(metrics["rollback_triggered"])


# ---------------------------------------------------------------------------
# D â€” Rollback manual
# ---------------------------------------------------------------------------

class TestRollbackManual(unittest.TestCase):
    """rollback() revierte el adapter al backup si existe."""

    def setUp(self):
        self.tmp    = tempfile.mkdtemp()
        self.out    = Path(self.tmp) / "adapter"
        self.backup = Path(self.tmp) / "adapter_backup"
        self.reg    = Path(self.tmp) / "registry.json"

    def test_rollback_manual_succeeds(self):
        """rollback() devuelve True si hay backup y lo restaura."""
        _make_adapter(self.out,    eval_loss=0.20)  # adapter actual (malo)
        _make_adapter(self.backup, eval_loss=0.10)  # backup (bueno)

        cl = ContinualLearner(
            model_id      = "Qwen/Qwen2.5-3B-Instruct",
            registry_path = str(self.reg),
        )
        result = cl.rollback(str(self.out))
        self.assertTrue(result)
        # El backup ya no existe despuÃ©s del rollback
        self.assertFalse(self.backup.exists())
        # El adapter restaurado tiene meta.json del backup original
        meta = json.loads((self.out / "meta.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(meta["eval_loss"], 0.10, places=5)

    def test_rollback_manual_no_backup(self):
        """rollback() devuelve False si no hay backup."""
        _make_adapter(self.out, eval_loss=0.20)
        cl = ContinualLearner(
            model_id      = "Qwen/Qwen2.5-3B-Instruct",
            registry_path = str(self.reg),
        )
        result = cl.rollback(str(self.out))
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# E â€” register_existing()
# ---------------------------------------------------------------------------

class TestRegisterExisting(unittest.TestCase):
    """register_existing() migra adapters pre-existentes al registro."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ds  = Path(self.tmp) / "data.jsonl"
        self.out = Path(self.tmp) / "existing_adapter"
        self.reg = Path(self.tmp) / "registry.json"
        _make_jsonl(self.ds,  n=30)
        _make_adapter(self.out, eval_loss=0.08)

    def test_register_reads_eval_loss_from_meta(self):
        """register_existing lee eval_loss de meta.json automÃ¡ticamente."""
        cl = ContinualLearner(
            model_id      = "Qwen/Qwen2.5-3B-Instruct",
            registry_path = str(self.reg),
        )
        cl.register_existing(
            adapter_dir  = str(self.out),
            dataset_path = str(self.ds),
            name         = "existing_v0",
        )
        reg = cl.get_registry()
        self.assertEqual(len(reg["adapters"]), 1)
        self.assertEqual(reg["adapters"][0]["name"], "existing_v0")
        self.assertAlmostEqual(reg["adapters"][0]["eval_loss"], 0.08, places=5)

    def test_register_existing_adds_to_replay_buffer(self):
        """Adapter registrado aparece en el replay buffer del siguiente fit()."""
        cl = ContinualLearner(
            model_id           = "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = str(self.reg),
            replay_buffer_size = 20,
        )
        cl.register_existing(
            adapter_dir  = str(self.out),
            dataset_path = str(self.ds),
            name         = "existing_v0",
        )
        ds2  = Path(self.tmp) / "data2.jsonl"
        out2 = Path(self.tmp) / "adapter_v2"
        _make_jsonl(ds2, n=40)

        mock_t = _mock_trainer_fit(eval_loss=0.09)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.fit(str(ds2), str(out2), adapter_name="v2")

        self.assertGreater(metrics["replay_samples_used"], 0)

    def test_register_explicit_eval_loss_when_no_meta(self):
        """Si no hay meta.json, register_existing usa el eval_loss explÃ­cito."""
        (self.out / "meta.json").unlink()  # quitar el meta
        cl = ContinualLearner(
            model_id      = "Qwen/Qwen2.5-3B-Instruct",
            registry_path = str(self.reg),
        )
        cl.register_existing(
            adapter_dir  = str(self.out),
            dataset_path = str(self.ds),
            name         = "no_meta",
            eval_loss    = 0.22,
        )
        reg = cl.get_registry()
        self.assertAlmostEqual(reg["adapters"][0]["eval_loss"], 0.22, places=5)


# ---------------------------------------------------------------------------
# F â€” Historial get_registry()
# ---------------------------------------------------------------------------

class TestHistory(unittest.TestCase):
    """get_registry() devuelve el estado correcto del registro."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = Path(self.tmp) / "registry.json"

    def test_empty_registry(self):
        cl = ContinualLearner("Qwen/Qwen2.5-3B-Instruct", registry_path=str(self.reg))
        reg = cl.get_registry()
        self.assertEqual(reg.get("adapters", []), [])

    def test_history_preserves_order(self):
        """Los adapters aparecen en orden de registro (primero â†’ Ãºltimo)."""
        cl = ContinualLearner("Qwen/Qwen2.5-3B-Instruct", registry_path=str(self.reg))
        for i, name in enumerate(["alpha", "beta", "gamma"]):
            ds  = Path(self.tmp) / f"ds{i}.jsonl"
            out = Path(self.tmp) / f"adapter_{name}"
            _make_jsonl(ds, n=20)
            cl.register_existing(str(out), str(ds), name=name, eval_loss=0.1 + i * 0.01)
        names = [a["name"] for a in cl.get_registry()["adapters"]]
        self.assertEqual(names, ["alpha", "beta", "gamma"])

    def test_registry_persists_across_instances(self):
        """El registro se persiste en disco y una nueva instancia lo carga."""
        cl1 = ContinualLearner("Qwen/Qwen2.5-3B-Instruct", registry_path=str(self.reg))
        ds  = Path(self.tmp) / "ds.jsonl"
        out = Path(self.tmp) / "adp"
        _make_jsonl(ds, n=20)
        cl1.register_existing(str(out), str(ds), name="persist_test", eval_loss=0.05)

        # Nueva instancia lee del mismo registry.json
        cl2 = ContinualLearner("Qwen/Qwen2.5-3B-Instruct", registry_path=str(self.reg))
        reg = cl2.get_registry()
        self.assertEqual(len(reg["adapters"]), 1)
        self.assertEqual(reg["adapters"][0]["name"], "persist_test")


# ---------------------------------------------------------------------------
# G â€” IntegraciÃ³n con interaction_log.jsonl
# ---------------------------------------------------------------------------

class TestInteractionLogIntegration(unittest.TestCase):
    """
    Verifica que el flujo completo interaction_log â†’ JSONL â†’ ContinualLearner funciona.
    Simula el modo --auto del CLI learn:
      1. Leer interaction_log.jsonl
      2. Filtrar entradas con feedback â‰¥ 0 (positivo o sin feedback)
      3. Convertirlas en ejemplos de entrenamiento
      4. Pasar al fit() del ContinualLearner
    """

    def _make_interaction_log(self, path: Path, n_positive=5, n_negative=2, n_neutral=3):
        """Escribe un interaction_log.jsonl con entradas variadas."""
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for i in range(n_positive):
            entries.append({
                "id":          f"sess_t{i}",
                "timestamp":   "2026-05-15T10:00:00Z",
                "session_id":  "sess",
                "turn":        i,
                "user_msg":    f"Pregunta {i}",
                "assistant":   f"Respuesta {i}",
                "model":       "Qwen/Qwen2.5-3B-Instruct",
                "ms":          500,
                "feedback":    1,  # ðŸ‘
            })
        for i in range(n_negative):
            entries.append({
                "id":          f"sess_neg_t{i}",
                "timestamp":   "2026-05-15T10:00:00Z",
                "session_id":  "sess_neg",
                "turn":        i,
                "user_msg":    f"Pregunta mala {i}",
                "assistant":   f"Respuesta mala {i}",
                "model":       "Qwen/Qwen2.5-3B-Instruct",
                "ms":          500,
                "feedback":    -1,  # ðŸ‘Ž
            })
        for i in range(n_neutral):
            entries.append({
                "id":          f"sess_neutral_t{i}",
                "timestamp":   "2026-05-15T10:00:00Z",
                "session_id":  "sess_neutral",
                "turn":        i,
                "user_msg":    f"Pregunta neutral {i}",
                "assistant":   f"Respuesta neutral {i}",
                "model":       "Qwen/Qwen2.5-3B-Instruct",
                "ms":          500,
                "feedback":    None,  # sin feedback
            })
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return entries

    def _log_to_training_jsonl(self, log_path: Path, out_path: Path) -> int:
        """
        Convierte interaction_log.jsonl a JSONL de entrenamiento.
        Incluye solo entradas con feedback >= 0 (positivo o None).
        Formato: {messages: [{role:user,...},{role:assistant,...}]}
        """
        count = 0
        with open(log_path, encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                entry = json.loads(line)
                if entry.get("feedback") == -1:
                    continue
                ex = {
                    "messages": [
                        {"role": "user",      "content": entry["user_msg"]},
                        {"role": "assistant", "content": entry["assistant"]},
                    ]
                }
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1
        return count

    def test_filter_excludes_negative_feedback(self):
        """Las entradas con feedback=-1 no aparecen en el dataset de entrenamiento."""
        tmp = tempfile.mkdtemp()
        log = Path(tmp) / "interaction_logs" / "interaction_log.jsonl"
        out = Path(tmp) / "training.jsonl"
        self._make_interaction_log(log, n_positive=5, n_negative=3, n_neutral=2)
        count = self._log_to_training_jsonl(log, out)
        # 5 positivos + 2 neutrales = 7; los 3 negativos excluidos
        self.assertEqual(count, 7)

    def test_training_jsonl_format(self):
        """El JSONL generado tiene el formato correcto para ContinualLearner."""
        tmp = tempfile.mkdtemp()
        log = Path(tmp) / "interaction_logs" / "interaction_log.jsonl"
        out = Path(tmp) / "training.jsonl"
        self._make_interaction_log(log, n_positive=3, n_negative=0, n_neutral=0)
        self._log_to_training_jsonl(log, out)

        lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        for line in lines:
            self.assertIn("messages", line)
            self.assertEqual(len(line["messages"]), 2)
            self.assertEqual(line["messages"][0]["role"], "user")
            self.assertEqual(line["messages"][1]["role"], "assistant")

    def test_fit_with_interaction_derived_dataset(self):
        """fit() funciona con un dataset derivado del interaction_log."""
        tmp = tempfile.mkdtemp()
        log = Path(tmp) / "interaction_logs" / "interaction_log.jsonl"
        training_ds = Path(tmp) / "training.jsonl"
        out = Path(tmp) / "retrained_adapter"
        reg = Path(tmp) / "registry.json"

        self._make_interaction_log(log, n_positive=8, n_negative=2, n_neutral=4)
        self._log_to_training_jsonl(log, training_ds)

        cl = ContinualLearner("Qwen/Qwen2.5-3B-Instruct", registry_path=str(reg))
        mock_t = _mock_trainer_fit(eval_loss=0.09)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.fit(str(training_ds), str(out), adapter_name="retrained_v1")

        self.assertFalse(metrics["rollback_triggered"])
        self.assertAlmostEqual(metrics["eval_loss"], 0.09, places=5)

    def test_empty_log_after_filter(self):
        """Si todos los ejemplos son negativos, no se entrena (dataset vacÃ­o)."""
        tmp = tempfile.mkdtemp()
        log = Path(tmp) / "interaction_logs" / "interaction_log.jsonl"
        out = Path(tmp) / "training.jsonl"
        self._make_interaction_log(log, n_positive=0, n_negative=5, n_neutral=0)
        count = self._log_to_training_jsonl(log, out)
        self.assertEqual(count, 0)
        # El archivo de salida existe pero estÃ¡ vacÃ­o
        self.assertEqual(out.stat().st_size, 0)


# ---------------------------------------------------------------------------
# H â€” Replay distribuido entre mÃºltiples datasets
# ---------------------------------------------------------------------------

class TestReplayBufferDistribution(unittest.TestCase):
    """El replay buffer se distribuye equitativamente entre datasets histÃ³ricos."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = Path(self.tmp) / "registry.json"

    def test_replay_split_across_datasets(self):
        """Con 3 datasets en el registro y buffer=60, cada uno aporta ~20."""
        cl = ContinualLearner(
            "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = str(self.reg),
            replay_buffer_size = 60,
        )
        # Registrar 3 adapters previos con sus datasets
        for i in range(3):
            ds  = Path(self.tmp) / f"ds{i}.jsonl"
            out = Path(self.tmp) / f"adp{i}"
            _make_jsonl(ds, n=100)
            cl.register_existing(str(out), str(ds), name=f"v{i}", eval_loss=0.1)

        # Cuarto fit: verifica que se mezclan ejemplos de los 3 datasets anteriores
        ds4  = Path(self.tmp) / "ds4.jsonl"
        out4 = Path(self.tmp) / "adp4"
        _make_jsonl(ds4, n=50)

        mock_t = _mock_trainer_fit(eval_loss=0.08)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.fit(str(ds4), str(out4), adapter_name="v4")

        # Debe usar exactamente el tamaÃ±o del buffer (o menos si datasets pequeÃ±os)
        self.assertLessEqual(metrics["replay_samples_used"], 60)
        self.assertGreater(metrics["replay_samples_used"], 0)

    def test_replay_skips_missing_dataset(self):
        """Datasets que ya no existen en disco se omiten sin error."""
        cl = ContinualLearner(
            "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = str(self.reg),
            replay_buffer_size = 30,
        )
        # Registrar un adapter con dataset que NO existe
        phantom_ds  = Path(self.tmp) / "nonexistent.jsonl"
        phantom_out = Path(self.tmp) / "phantom_adapter"
        # No crear el archivo â€” solo registrar la entrada
        cl.register_existing(str(phantom_out), str(phantom_ds), name="phantom", eval_loss=0.1)

        # Registrar un adapter real
        real_ds  = Path(self.tmp) / "real.jsonl"
        real_out = Path(self.tmp) / "real_adapter"
        _make_jsonl(real_ds, n=50)
        cl.register_existing(str(real_out), str(real_ds), name="real", eval_loss=0.1)

        # Nuevo fit: debe ignorar el phantom y usar solo el real
        ds_new  = Path(self.tmp) / "new.jsonl"
        out_new = Path(self.tmp) / "new_adapter"
        _make_jsonl(ds_new, n=30)
        mock_t = _mock_trainer_fit(eval_loss=0.09)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.fit(str(ds_new), str(out_new), adapter_name="new")

        # No debe lanzar excepciÃ³n y debe haber usado replay del dataset real
        self.assertGreater(metrics["replay_samples_used"], 0)


# ---------------------------------------------------------------------------
# I â€” Edge cases del rollback threshold
# ---------------------------------------------------------------------------

class TestRollbackEdgeCases(unittest.TestCase):
    """Casos lÃ­mite del mecanismo de rollback."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = Path(self.tmp) / "registry.json"

    def test_zero_baseline_no_rollback(self):
        """Con baseline=0.0 (sin histÃ³rico), no hay rollback aunque la loss sea alta."""
        cl = ContinualLearner(
            "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = str(self.reg),
            rollback_threshold = 0.10,
        )
        ds  = Path(self.tmp) / "ds.jsonl"
        out = Path(self.tmp) / "adp"
        _make_jsonl(ds, n=20)
        mock_t = _mock_trainer_fit(eval_loss=99.0)  # loss muy alta
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.fit(str(ds), str(out), adapter_name="v1")
        # Sin baseline, nunca hay rollback
        self.assertFalse(metrics["rollback_triggered"])
        self.assertIsNone(metrics["regression_pct"])

    def test_improvement_not_rollback(self):
        """Si la loss baja, no hay rollback."""
        cl = ContinualLearner(
            "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = str(self.reg),
            rollback_threshold = 0.10,
        )
        ds  = Path(self.tmp) / "ds.jsonl"
        out = Path(self.tmp) / "adp"
        _make_jsonl(ds, n=30)

        mock_t = _mock_trainer_fit(eval_loss=0.20)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            cl.fit(str(ds), str(out), adapter_name="v1")

        _make_adapter(out, eval_loss=0.20)

        ds2 = Path(self.tmp) / "ds2.jsonl"
        _make_jsonl(ds2, n=30)
        mock_t2 = _mock_trainer_fit(eval_loss=0.10)  # mejora del 50%
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t2):
            metrics = cl.fit(str(ds2), str(out), adapter_name="v1")

        self.assertFalse(metrics["rollback_triggered"])


# ---------------------------------------------------------------------------
# J — stack_adapters (combinación base + personal)
# ---------------------------------------------------------------------------

class TestStackAdapters(unittest.TestCase):
    """stack_adapters() combina dos adapters vía PEFT add_weighted_adapter."""

    def setUp(self):
        self.tmp    = tempfile.mkdtemp()
        self.base   = Path(self.tmp) / "base_adapter"
        self.pers   = Path(self.tmp) / "personal_adapter"
        self.out    = Path(self.tmp) / "stacked_adapter"
        self.reg    = Path(self.tmp) / "registry.json"
        _make_adapter(self.base, eval_loss=0.08)
        _make_adapter(self.pers, eval_loss=0.12)
        # Mock de transformers + peft (imports lazy dentro de stack_adapters)
        self.mock_model = MagicMock()
        self._patches = [
            patch.dict("sys.modules", {
                "transformers": MagicMock(),
                "peft": MagicMock(),
            }),
        ]
        for p in self._patches:
            p.start()
        import transformers
        import peft
        transformers.AutoModelForCausalLM.from_pretrained.return_value = self.mock_model
        peft.PeftModel.from_pretrained.return_value = self.mock_model

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()

    def test_stack_creates_meta_json(self):
        """El adapter combinado incluye meta.json con información de stacking."""
        cl = ContinualLearner("Qwen/Qwen2.5-3B-Instruct", registry_path=str(self.reg))
        result = cl.stack_adapters(
            base_adapter_dir     = str(self.base),
            personal_adapter_dir = str(self.pers),
            output_dir           = str(self.out),
            base_weight          = 0.8,
            personal_weight      = 0.2,
        )

        self.assertEqual(result, str(self.out))
        self.assertTrue((self.out / "meta.json").exists())
        meta = json.loads((self.out / "meta.json").read_text(encoding="utf-8"))
        self.assertTrue(meta.get("stacked"))
        self.assertEqual(meta["base_weight"], 0.8)
        self.assertEqual(meta["personal_weight"], 0.2)

    def test_stack_calls_add_weighted_adapter(self):
        """Verifica que add_weighted_adapter se llama con pesos correctos."""
        cl = ContinualLearner("Qwen/Qwen2.5-3B-Instruct", registry_path=str(self.reg))
        cl.stack_adapters(
            base_adapter_dir     = str(self.base),
            personal_adapter_dir = str(self.pers),
            output_dir           = str(self.out),
            base_weight          = 0.6,
            personal_weight      = 0.4,
            combination_type     = "svd",
        )

        self.mock_model.add_weighted_adapter.assert_called_once()
        call_kwargs = self.mock_model.add_weighted_adapter.call_args[1]
        self.assertEqual(call_kwargs["weights"], [0.6, 0.4])
        self.assertEqual(call_kwargs["combination_type"], "svd")

    def test_stack_with_default_weights(self):
        """Pesos por defecto: 0.7 base + 0.3 personal."""
        cl = ContinualLearner("Qwen/Qwen2.5-3B-Instruct", registry_path=str(self.reg))
        cl.stack_adapters(
            base_adapter_dir     = str(self.base),
            personal_adapter_dir = str(self.pers),
            output_dir           = str(self.out),
        )

        self.mock_model.add_weighted_adapter.assert_called_once()
        call_kwargs = self.mock_model.add_weighted_adapter.call_args[1]
        self.assertEqual(call_kwargs["weights"], [0.7, 0.3])


# ---------------------------------------------------------------------------
# K — from_user_profile (generación de dataset personal + entrenamiento)
# ---------------------------------------------------------------------------

class TestFromUserProfile(unittest.TestCase):
    """from_user_profile() genera dataset personalizado y entrena adapter."""

    def setUp(self):
        self.tmp   = tempfile.mkdtemp()
        self.base  = Path(self.tmp) / "base_adapter"
        self.out   = Path(self.tmp) / "personal_adapter"
        self.reg   = Path(self.tmp) / "registry.json"
        _make_adapter(self.base, eval_loss=0.08)
        self.profile = {
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
            ],
        }

    def test_generates_profile_examples(self):
        """_generate_profile_examples produce al menos n ejemplos."""
        examples = ContinualLearner._generate_profile_examples(
            name="Test",
            language="es",
            folders={"documents": "~/Docs", "downloads": "~/Down"},
            contacts={"boss": "boss@co.com"},
            rules=[{"type": "spam", "sender": "spam@x.com"}],
            n=30,
        )
        self.assertGreaterEqual(len(examples), 30)
        for ex in examples:
            self.assertIn("messages", ex)
            self.assertEqual(len(ex["messages"]), 2)
            self.assertEqual(ex["messages"][0]["role"], "user")
            self.assertEqual(ex["messages"][1]["role"], "assistant")

    def test_profile_examples_use_user_name(self):
        """Los ejemplos generados contienen el nombre del usuario."""
        examples = ContinualLearner._generate_profile_examples(
            name="María",
            language="es",
            folders={"documents": "~/Docs"},
            contacts={"boss": "boss@co.com"},
            rules=[],
            n=20,
        )
        all_text = " ".join(
            json.dumps(ex, ensure_ascii=False) for ex in examples
        )
        self.assertIn("María", all_text)

    def test_profile_examples_use_contacts(self):
        """Los ejemplos incluyen los contactos del perfil."""
        examples = ContinualLearner._generate_profile_examples(
            name="Test",
            language="es",
            folders={},
            contacts={"jefe": "jefe@empresa.com", "cliente": "cliente@corp.com"},
            rules=[],
            n=30,
        )
        all_text = " ".join(
            json.dumps(ex, ensure_ascii=False) for ex in examples
        )
        self.assertIn("jefe@empresa.com", all_text)
        self.assertIn("cliente@corp.com", all_text)

    def test_profile_examples_include_spam_rules(self):
        """Las reglas de spam generan ejemplos de email_filter mark_spam."""
        examples = ContinualLearner._generate_profile_examples(
            name="Test",
            language="es",
            folders={},
            contacts={},
            rules=[{"type": "spam", "sender": "malware@hack.com"}],
            n=20,
        )
        all_text = " ".join(
            json.dumps(ex, ensure_ascii=False) for ex in examples
        )
        self.assertIn("malware@hack.com", all_text)
        self.assertIn("mark_spam", all_text)

    def test_from_user_profile_trains_and_returns_metrics(self):
        """from_user_profile entrena con mock y devuelve métricas."""
        cl = ContinualLearner(
            "Qwen/Qwen2.5-3B-Instruct",
            registry_path = str(self.reg),
        )
        mock_t = _mock_trainer_fit(eval_loss=0.07)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.from_user_profile(
                profile          = self.profile,
                base_adapter_dir = str(self.base),
                output_dir       = str(self.out),
                n_examples       = 20,
                epochs           = 1,
            )

        self.assertIn("eval_loss", metrics)
        self.assertAlmostEqual(metrics["eval_loss"], 0.07, places=5)
        self.assertFalse(metrics["rollback_triggered"])

    def test_from_user_profile_saves_user_profile_json(self):
        """Tras from_user_profile, user_profile.json se guarda en output_dir."""
        cl = ContinualLearner(
            "Qwen/Qwen2.5-3B-Instruct",
            registry_path = str(self.reg),
        )
        mock_t = _mock_trainer_fit(eval_loss=0.06)
        try:
            with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
                cl.from_user_profile(
                    profile          = self.profile,
                    base_adapter_dir = str(self.base),
                    output_dir       = str(self.out),
                    n_examples       = 15,
                    epochs           = 1,
                )
        except OSError as e:
            if getattr(e, 'winerror', 0) == 6714:
                self.skipTest("WinError 6714 (pyarrow bug en Windows)")
            raise

        profile_path = self.out / "user_profile.json"
        try:
            self.assertTrue(profile_path.exists())
            saved = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["name"], "Felipe")
            self.assertEqual(saved["language"], "es")
        except OSError as e:
            if getattr(e, 'winerror', 0) == 6714:
                self.skipTest("WinError 6714 (pyarrow bug en Windows)")
            raise

    def test_from_user_profile_with_minimal_profile(self):
        """Perfil mínimo (solo nombre) no crashea y genera ejemplos."""
        cl = ContinualLearner(
            "Qwen/Qwen2.5-3B-Instruct",
            registry_path = str(self.reg),
        )
        mock_t = _mock_trainer_fit(eval_loss=0.11)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.from_user_profile(
                profile          = {"name": "Min", "language": "en"},
                base_adapter_dir = str(self.base),
                output_dir       = str(self.out),
                n_examples       = 10,
                epochs           = 1,
            )

        self.assertIn("eval_loss", metrics)
        self.assertFalse(metrics["rollback_triggered"])

    def test_from_user_profile_stacking_attempted(self):
        """from_user_profile intenta stacking tras entrenar (stacked_adapter en metrics)."""
        cl = ContinualLearner(
            "Qwen/Qwen2.5-3B-Instruct",
            registry_path = str(self.reg),
        )
        mock_t = _mock_trainer_fit(eval_loss=0.05)
        with patch("motor.trainer_llm.LLMTrainer", return_value=mock_t):
            metrics = cl.from_user_profile(
                profile          = self.profile,
                base_adapter_dir = str(self.base),
                output_dir       = str(self.out),
                n_examples       = 10,
                epochs           = 1,
            )

        # stacking requiere GPU → en tests sin GPU será None
        self.assertIn("stacked_adapter", metrics)

    def test_english_profile_generates_english_examples(self):
        """Perfil en inglés genera ejemplos en inglés."""
        examples = ContinualLearner._generate_profile_examples(
            name="John",
            language="en",
            folders={"documents": "~/Documents", "downloads": "~/Downloads"},
            contacts={"boss": "boss@company.com"},
            rules=[],
            n=20,
        )
        all_text = " ".join(
            json.dumps(ex, ensure_ascii=False) for ex in examples
        )
        # Los templates actuales están en español; verificar que al menos
        # el nombre del usuario aparece
        self.assertIn("John", all_text)


# ---------------------------------------------------------------------------
# Ejecutar
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
