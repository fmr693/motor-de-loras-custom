"""
tests/test_gemma4_support.py
============================
Soporte de la arquitectura gemma4_unified (Gemma 4 12B-it) en el pipeline
de entrenamiento (sesión 11-jun-2026).

Datos reales del config.json de google/gemma-4-12B-it (verificados en vivo):
  model_type: "gemma4_unified"
  architectures: ["Gemma4UnifiedForConditionalGeneration"]
  claves: text_config (con attention_k_eq_v), vision_config, audio_config

Cubre:
  1. ModelAnalyzer reconoce gemma4_unified: familia gemma4, engine peft_trl,
     is_vlm=False EXPLÍCITO aunque tenga vision_config (entrena como texto)
  2. La heurística VLM clásica sigue intacta para otras familias
  3. LLMTrainer._detect_multimodal_lm() vía config.json crudo (funciona
     aunque transformers no conozca la arquitectura)
  4. LLMTrainer._filter_target_modules() ajusta los nombres a los módulos
     reales del modelo (KV unificado → k_proj/v_proj pueden no existir)
"""
from __future__ import annotations

import json

import pytest

from motor.analyzer import ModelAnalyzer
from motor.trainer_llm import LLMTrainer


# Config mínimo imitando el real de google/gemma-4-12B-it
GEMMA4_CONFIG = {
    "model_type": "gemma4_unified",
    "architectures": ["Gemma4UnifiedForConditionalGeneration"],
    "text_config": {
        "model_type": "gemma4_unified_text",
        "attention_k_eq_v": True,
        "hidden_size": 3840,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "intermediate_size": 15360,
        "vocab_size": 262144,
    },
    "vision_config": {"hidden_size": 1152},
    "audio_config": {"hidden_size": 1536},
}


@pytest.fixture()
def gemma4_dir(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(GEMMA4_CONFIG), encoding="utf-8"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# 1. ModelAnalyzer reconoce gemma4_unified
# ---------------------------------------------------------------------------

class TestAnalyzerGemma4:
    def test_familia_y_engine(self, gemma4_dir):
        result = ModelAnalyzer(str(gemma4_dir)).analyze()
        assert result.family == "gemma4"
        assert result.model_type == "gemma4_unified"
        assert result.engine == "peft_trl"
        assert "q_proj" in result.target_modules

    def test_no_es_vlm_aunque_tenga_vision_config(self, gemma4_dir):
        """gemma4_unified se entrena como texto 'in one pass' (model card):
        el is_vlm=False explícito de la familia manda sobre la heurística."""
        result = ModelAnalyzer(str(gemma4_dir)).analyze()
        assert result.is_vlm is False

    def test_no_avisa_familia_desconocida(self, gemma4_dir):
        result = ModelAnalyzer(str(gemma4_dir)).analyze()
        assert not any("no reconocida" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 2. La heurística VLM clásica sigue intacta
# ---------------------------------------------------------------------------

class TestVlmHeuristicaIntacta:
    def test_familia_desconocida_con_vision_config_es_vlm(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "futuro_modelo_x",
            "vision_config": {"hidden_size": 64},
        }), encoding="utf-8")
        result = ModelAnalyzer(str(tmp_path)).analyze()
        assert result.is_vlm is True

    def test_qwen2_vl_sigue_siendo_vlm(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen2_vl",
            "vision_config": {"hidden_size": 64},
        }), encoding="utf-8")
        result = ModelAnalyzer(str(tmp_path)).analyze()
        assert result.is_vlm is True


# ---------------------------------------------------------------------------
# 3. Detección multimodal del trainer (config crudo, sin AutoConfig)
# ---------------------------------------------------------------------------

def _bare_trainer(model_id: str) -> LLMTrainer:
    """LLMTrainer sin pasar por __init__ (evita detección de hardware)."""
    t = object.__new__(LLMTrainer)
    t.model_id  = model_id
    t.cache_dir = None
    return t


class TestDeteccionMultimodal:
    def test_gemma4_unified_detectado(self, gemma4_dir):
        assert _bare_trainer(str(gemma4_dir))._detect_multimodal_lm() is True

    def test_qwen2_no_es_multimodal_lm(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "qwen2"}), encoding="utf-8"
        )
        assert _bare_trainer(str(tmp_path))._detect_multimodal_lm() is False

    def test_sin_config_no_explota(self, tmp_path):
        assert _bare_trainer(str(tmp_path))._detect_multimodal_lm() is False


# ---------------------------------------------------------------------------
# 4. Filtro de target_modules contra el modelo real
# ---------------------------------------------------------------------------

class TestFiltroTargetModules:
    @pytest.fixture()
    def modelo_kv_unificado(self):
        """Modelo de juguete imitando KV unificado: kv_proj en vez de k/v."""
        import torch.nn as nn

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj  = nn.Linear(4, 4)
                self.kv_proj = nn.Linear(4, 4)   # ← fusionado
                self.o_proj  = nn.Linear(4, 4)
                self.norm    = nn.LayerNorm(4)   # no-Linear: debe ignorarse

        return Tiny()

    def test_descarta_los_que_no_existen(self, modelo_kv_unificado):
        kept = LLMTrainer._filter_target_modules(
            modelo_kv_unificado,
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        assert kept == ["q_proj", "o_proj"]

    def test_autodetecta_si_ninguno_coincide(self, modelo_kv_unificado):
        kept = LLMTrainer._filter_target_modules(
            modelo_kv_unificado, ["query", "value"],
        )
        # Autodetección: todas las *_proj reales (incluida la fusionada)
        assert set(kept) == {"q_proj", "kv_proj", "o_proj"}

    def test_todos_existen_no_cambia_nada(self, modelo_kv_unificado):
        kept = LLMTrainer._filter_target_modules(
            modelo_kv_unificado, ["q_proj", "o_proj"],
        )
        assert kept == ["q_proj", "o_proj"]
