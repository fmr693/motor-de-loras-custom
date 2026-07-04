"""
tests/test_dpo_pipeline.py
==========================
Tests unitarios para DPOBuilder (motor.dpo_trainer) — S10.4.
No requieren GPU. El DPOTrainer de TRL se mockea completamente.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from motor.dpo_trainer import DPOBuilder, _find_original_prompt, _load_pairs_from_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log(path: Path, entries: list[dict]) -> Path:
    """Escribe una lista de entradas en un JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


def _entry(user_msg: str, assistant: str, feedback, interaction_id: str = "x") -> dict:
    return {
        "id":        interaction_id,
        "timestamp": "2024-01-01T00:00:00Z",
        "user_msg":  user_msg,
        "assistant": assistant,
        "feedback":  feedback,
        "session_id": "s1",
        "turn":       1,
        "model":      "test",
        "ms":         100,
    }


# ---------------------------------------------------------------------------
# TestLoadEntries
# ---------------------------------------------------------------------------

class TestLoadEntries(unittest.TestCase):
    """DPOBuilder.load_entries() filtra solo entradas con feedback."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.log = Path(self.tmp) / "log.jsonl"

    def test_loads_only_rated(self):
        _make_log(self.log, [
            _entry("Q1", "A_pos", 1),
            _entry("Q1", "A_neg", -1),
            _entry("Q2", "A_no",  None),   # sin feedback — no incluido
        ])
        b = DPOBuilder(self.log)
        entries = b.load_entries()
        self.assertEqual(len(entries), 2)

    def test_empty_log(self):
        _make_log(self.log, [])
        b = DPOBuilder(self.log)
        self.assertEqual(b.load_entries(), [])

    def test_missing_log_raises(self):
        b = DPOBuilder(Path(self.tmp) / "missing.jsonl")
        with self.assertRaises(FileNotFoundError):
            b.load_entries()

    def test_invalid_json_skipped_not_fatal(self):
        # Cambio 12-jun-2026: una línea corrupta entre miles ya NO aborta el
        # DPO (un log lo escribe el servidor 24/7); se salta y se cuenta.
        self.log.write_text("not-json\n", encoding="utf-8")
        b = DPOBuilder(self.log)
        self.assertEqual(b.load_entries(), [])

    def test_invalid_json_no_descarta_las_buenas(self):
        # Una línea mala no debe tirar las entradas válidas que la rodean.
        good = json.dumps({"user_msg": "hola que tal estas hoy", "assistant":
                           "Muy bien, gracias por preguntar.", "feedback": 1})
        self.log.write_text(good + "\nnot-json\n" + good.replace("hola", "adios") +
                            "\n", encoding="utf-8")
        b = DPOBuilder(self.log)
        self.assertEqual(len(b.load_entries()), 2)


# ---------------------------------------------------------------------------
# TestBuildPairs
# ---------------------------------------------------------------------------

class TestBuildPairs(unittest.TestCase):
    """DPOBuilder.build_pairs() forma pares chosen/rejected correctamente."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.log = Path(self.tmp) / "log.jsonl"

    def test_single_pair(self):
        _make_log(self.log, [
            _entry("Q1", "Buena respuesta",  1),
            _entry("Q1", "Mala respuesta",  -1),
        ])
        b = DPOBuilder(self.log, min_pairs=1)
        pairs = b.build_pairs()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"],   "Buena respuesta")
        self.assertEqual(pairs[0]["rejected"], "Mala respuesta")

    def test_prompt_preserved(self):
        _make_log(self.log, [
            _entry("¿Cuál es la capital?", "París",  1),
            _entry("¿Cuál es la capital?", "Madrid", -1),
        ])
        b = DPOBuilder(self.log, min_pairs=1)
        pairs = b.build_pairs()
        self.assertEqual(pairs[0]["prompt"], "¿Cuál es la capital?")

    def test_no_pair_without_both_ratings(self):
        _make_log(self.log, [
            _entry("Q1", "Bien", 1),
            _entry("Q2", "Mal",  -1),   # prompts distintos — sin par
        ])
        b = DPOBuilder(self.log, min_pairs=1)
        pairs = b.build_pairs()
        self.assertEqual(len(pairs), 0)

    def test_identical_responses_skipped(self):
        _make_log(self.log, [
            _entry("Q1", "Misma respuesta", 1),
            _entry("Q1", "Misma respuesta", -1),
        ])
        b = DPOBuilder(self.log, min_pairs=1)
        pairs = b.build_pairs()
        self.assertEqual(len(pairs), 0)

    def test_multiple_pairs_same_prompt(self):
        """2 positivos × 2 negativos = 4 pares (sin repetidos)."""
        _make_log(self.log, [
            _entry("Q",  "A+1",  1, "i1"),
            _entry("Q",  "A+2",  1, "i2"),
            _entry("Q",  "A-1", -1, "i3"),
            _entry("Q",  "A-2", -1, "i4"),
        ])
        b = DPOBuilder(self.log, min_pairs=1)
        pairs = b.build_pairs()
        self.assertEqual(len(pairs), 4)

    def test_normalize_prompt(self):
        """Prompts con distinto case/espacios se agrupan si normalize=True."""
        _make_log(self.log, [
            _entry("Hola mundo",   "Buena",  1),
            _entry("  hola mundo", "Mala",  -1),
        ])
        b = DPOBuilder(self.log, min_pairs=1, normalize_prompt=True)
        pairs = b.build_pairs()
        self.assertEqual(len(pairs), 1)

    def test_no_normalize_prompt(self):
        """Con normalize=False, prompts distintos no se agrupan."""
        _make_log(self.log, [
            _entry("Hola mundo",   "Buena",  1),
            _entry("  hola mundo", "Mala",  -1),
        ])
        b = DPOBuilder(self.log, min_pairs=1, normalize_prompt=False)
        pairs = b.build_pairs()
        self.assertEqual(len(pairs), 0)


# ---------------------------------------------------------------------------
# TestToJsonl
# ---------------------------------------------------------------------------

class TestToJsonl(unittest.TestCase):
    """DPOBuilder.to_jsonl() escribe y valida pares mínimos."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.log = Path(self.tmp) / "log.jsonl"
        self.out = Path(self.tmp) / "pairs.jsonl"

    def test_writes_jsonl(self):
        _make_log(self.log, [
            _entry("Q", "Bien",  1),
            _entry("Q", "Mal",  -1),
        ])
        b = DPOBuilder(self.log, min_pairs=1)
        b.build_pairs()
        path = b.to_jsonl(self.out)
        self.assertTrue(path.exists())
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)
        self.assertIn("prompt",   lines[0])
        self.assertIn("chosen",   lines[0])
        self.assertIn("rejected", lines[0])

    def test_raises_below_min_pairs(self):
        _make_log(self.log, [
            _entry("Q1", "A", 1),
            _entry("Q1", "B", -1),
        ])
        b = DPOBuilder(self.log, min_pairs=5)
        b.build_pairs()
        with self.assertRaises(ValueError):
            b.to_jsonl(self.out)

    def test_creates_parent_dir(self):
        nested_out = Path(self.tmp) / "subdir" / "pairs.jsonl"
        _make_log(self.log, [
            _entry("Q", "Bien",  1),
            _entry("Q", "Mal",  -1),
        ])
        b = DPOBuilder(self.log, min_pairs=1)
        b.build_pairs()
        b.to_jsonl(nested_out)
        self.assertTrue(nested_out.exists())


# ---------------------------------------------------------------------------
# TestStats
# ---------------------------------------------------------------------------

class TestStats(unittest.TestCase):
    """DPOBuilder.stats() devuelve conteos correctos."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.log = Path(self.tmp) / "log.jsonl"

    def test_stats_counts(self):
        _make_log(self.log, [
            _entry("Q1", "A",  1),
            _entry("Q1", "B", -1),
            _entry("Q2", "C",  1),
        ])
        b = DPOBuilder(self.log, min_pairs=1)
        s = b.stats()
        self.assertEqual(s["rated_entries"], 3)
        self.assertEqual(s["positive"],      2)
        self.assertEqual(s["negative"],      1)
        self.assertEqual(s["pairs_available"], 1)  # solo Q1 tiene ambos

    def test_stats_no_pairs(self):
        _make_log(self.log, [
            _entry("Q1", "A", 1),
        ])
        b = DPOBuilder(self.log, min_pairs=1)
        s = b.stats()
        self.assertEqual(s["pairs_available"], 0)


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    """Funciones auxiliares internas."""

    def test_find_original_prompt(self):
        entries = [
            {"user_msg": "¿Hola Mundo?", "feedback": 1},
        ]
        result = _find_original_prompt(entries, "¿hola mundo?", normalize=True)
        self.assertEqual(result, "¿Hola Mundo?")

    def test_find_original_prompt_fallback(self):
        result = _find_original_prompt([], "fallback_key", normalize=True)
        self.assertEqual(result, "fallback_key")

    def test_load_pairs_from_jsonl(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "pairs.jsonl"
        pairs = [
            {"prompt": "P", "chosen": "C", "rejected": "R"},
        ]
        with open(tmp, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        loaded = _load_pairs_from_jsonl(tmp)
        self.assertEqual(loaded, pairs)


# ---------------------------------------------------------------------------
# TestFitMocked
# ---------------------------------------------------------------------------

class TestFitMocked(unittest.TestCase):
    """DPOBuilder.fit() con DPOTrainer completamente mockeado (sin GPU)."""

    def setUp(self):
        import tempfile
        self.tmp  = tempfile.mkdtemp()
        self.log  = Path(self.tmp) / "log.jsonl"
        self.out  = Path(self.tmp) / "adapter_dpo"

        _make_log(self.log, [
            _entry("Q1", "Buena respuesta",  1),
            _entry("Q1", "Mala respuesta",  -1),
        ])

    def _mock_fit_imports(self):
        """Context manager que parchea todos los imports lazy dentro de fit()."""
        import unittest.mock as _mock
        import sys

        # Crear módulos mock para las dependencias pesadas
        mock_torch     = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.bfloat16 = "bfloat16"

        mock_tok = MagicMock(); mock_tok.pad_token = None
        mock_tokenizer_cls = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tok

        mock_model = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_trainer_inst = MagicMock()
        mock_trainer_cls = MagicMock(return_value=mock_trainer_inst)

        mock_dataset_cls = MagicMock()
        mock_dataset_cls.from_list.return_value = MagicMock()

        mock_peft_model = MagicMock()

        patches = {
            "torch":                    mock_torch,
            "transformers":             MagicMock(
                AutoModelForCausalLM = mock_model_cls,
                AutoTokenizer        = mock_tokenizer_cls,
            ),
            "datasets":                 MagicMock(Dataset=mock_dataset_cls),
            "peft":                     MagicMock(
                LoraConfig       = MagicMock(),
                get_peft_model   = MagicMock(return_value=mock_model),
                TaskType         = MagicMock(CAUSAL_LM="CAUSAL_LM"),
            ),
            "trl":                      MagicMock(
                DPOTrainer  = mock_trainer_cls,
                DPOConfig   = MagicMock(),
                ORPOTrainer = mock_trainer_cls,
                ORPOConfig  = MagicMock(),
            ),
        }

        return patches, mock_trainer_inst, mock_model

    def test_fit_creates_meta_json(self):
        patches, _, _ = self._mock_fit_imports()
        with patch.dict("sys.modules", patches):
            b = DPOBuilder(self.log, min_pairs=1)
            b.fit(output_dir=self.out, base_model_id="test/model")

        meta_path = self.out / "meta.json"
        self.assertTrue(meta_path.exists(), "meta.json debe existir")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["training"], "orpo")
        self.assertEqual(meta["pairs"],    1)
        self.assertIn("train_time_s", meta)

    def test_fit_raises_below_min_pairs(self):
        _make_log(self.log, [
            _entry("Solo", "Pos", 1),   # sin negativo → sin par
        ])
        patches, _, _ = self._mock_fit_imports()
        b = DPOBuilder(self.log, min_pairs=1)
        # No llega a los imports — falla antes por falta de pares
        with self.assertRaises(ValueError):
            with patch.dict("sys.modules", patches):
                b.fit(output_dir=self.out, base_model_id="test/model")

    def test_fit_calls_trainer_train(self):
        patches, mock_trainer_inst, _ = self._mock_fit_imports()
        with patch.dict("sys.modules", patches):
            b = DPOBuilder(self.log, min_pairs=1)
            b.fit(output_dir=self.out, base_model_id="test/model")
        mock_trainer_inst.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
