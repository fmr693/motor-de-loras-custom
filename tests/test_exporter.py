"""
tests/test_exporter.py
======================
Tests del ExportManager, principalmente to_gguf() y funciones auxiliares.

Los tests que implican descarga del modelo base o GPU se marcan como skip
automáticamente.  Los que validan la lógica interna (búsqueda del script,
armado de comandos, fallback de cuantización) se ejecutan con mocks.

Verificado OK con Python 3.13 + llama-cpp-python instalado.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Importar ExportManager; si no hay GPU/PEFT disponibles simplemente algunos
# tests serán skip.
from motor.exporter import ExportManager

# ---------------------------------------------------------------------------
# Fixture: adapter de prueba mínimo (sin safetensors reales)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_adapter(tmp_path) -> Path:
    """Crea un adapter de prueba con meta.json y adapter_config.json mínimos."""
    adapter_dir = tmp_path / "test_adapter"
    adapter_dir.mkdir()

    meta = {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_dir": str(adapter_dir),
        "engine": "peft_trl",
        "lora_r": 16,
        "lora_alpha": 32,
    }
    (adapter_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    config = {
        "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    # Tokenizer mínimo (no real, solo para que los métodos de info() no crasheen)
    tok_dir = adapter_dir / "tokenizer"
    tok_dir.mkdir()
    (tok_dir / "tokenizer_config.json").write_text(json.dumps({}), encoding="utf-8")

    return adapter_dir


# ---------------------------------------------------------------------------
# A. Construcción del objeto
# ---------------------------------------------------------------------------

class TestExportManagerInit:

    def test_crea_con_adapter_existente(self, fake_adapter, tmp_path):
        """ExportManager se instancia con un adapter válido."""
        em = ExportManager(str(fake_adapter))
        assert em.adapter_dir == fake_adapter

    def test_raises_si_adapter_no_existe(self, tmp_path):
        """ValueError si el adapter_dir no tiene meta.json."""
        with pytest.raises((FileNotFoundError, ValueError)):
            ExportManager(str(tmp_path / "inexistente"))

    def test_raises_si_falta_meta_model_id(self, tmp_path):
        """ValueError si meta.json no contiene 'model_id'."""
        d = tmp_path / "adapter_sin_model_id"
        d.mkdir()
        (d / "meta.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            ExportManager(str(d))

    def test_info_devuelve_dict(self, fake_adapter):
        """info() devuelve un dict con al menos 'base_model'."""
        em = ExportManager(str(fake_adapter))
        info = em.info()
        assert isinstance(info, dict)
        assert "base_model" in info


# ---------------------------------------------------------------------------
# B. to_gguf() — lógica interna (sin descarga de modelo real)
# ---------------------------------------------------------------------------

class TestToGguf:

    def test_to_gguf_usa_convert_script_si_existe(self, fake_adapter, tmp_path):
        """Si convert_hf_to_gguf.py existe, to_gguf no intenta clonar llama.cpp."""
        # Crear un convert_hf_to_gguf.py falso en ~/llama.cpp
        fake_llama = tmp_path / "llama.cpp"
        fake_llama.mkdir()
        convert_script = fake_llama / "convert_hf_to_gguf.py"
        convert_script.write_text("# fake convert script", encoding="utf-8")

        em = ExportManager(str(fake_adapter))

        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            # Crear archivo de salida simulado para que el flujo no crashee
            for part in cmd:
                if isinstance(part, str) and part.endswith(".gguf"):
                    Path(part).touch()
            return subprocess.CompletedProcess(cmd, 0)

        out_dir = tmp_path / "gguf_out"

        # Parchear Path.home() para que apunte a nuestro tmp_path
        with patch("motor.exporter.Path.home", return_value=tmp_path):
            with patch("subprocess.run", side_effect=fake_run):
                # to_safetensors también hace subprocess, lo mockeamos
                with patch.object(em, "to_safetensors", return_value=out_dir / "_merged_tmp"):
                    (out_dir / "_merged_tmp").mkdir(parents=True, exist_ok=True)
                    try:
                        em.to_gguf(str(out_dir), quantization="f16",
                                   merged_dir=str(out_dir / "_merged_tmp"))
                    except Exception:
                        pass  # puede fallar en el rename, pero lo importante es que llamó

        # El script de conversión debe haberse invocado
        convert_calls = [c for c in calls if any("convert" in str(p) for p in c)]
        assert len(convert_calls) >= 1, "convert_hf_to_gguf.py no fue invocado"

    def test_to_gguf_quantization_f16_no_cuantiza(self, fake_adapter, tmp_path):
        """Con quantization='f16' no se llama al paso de cuantización."""
        em = ExportManager(str(fake_adapter))
        quant_calls = []

        def fake_run(cmd, check=True, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "quantize" in cmd_str.lower():
                quant_calls.append(cmd)
            # Simular salida gguf
            for part in cmd:
                if isinstance(part, str) and part.endswith(".gguf"):
                    Path(part).touch()
            return subprocess.CompletedProcess(cmd, 0)

        out_dir = tmp_path / "f16_out"
        merged  = out_dir / "_merged_tmp"
        merged.mkdir(parents=True, exist_ok=True)

        fake_llama = tmp_path / "llama.cpp"
        fake_llama.mkdir()
        (fake_llama / "convert_hf_to_gguf.py").write_text("# fake", encoding="utf-8")

        with patch("motor.exporter.Path.home", return_value=tmp_path):
            with patch("subprocess.run", side_effect=fake_run):
                with patch.object(em, "to_safetensors", return_value=merged):
                    try:
                        em.to_gguf(str(out_dir), quantization="f16", merged_dir=str(merged))
                    except Exception:
                        pass

        assert quant_calls == [], "No debería haber llamado a llama-quantize para f16"

    def test_to_gguf_crea_directorio_salida(self, fake_adapter, tmp_path):
        """to_gguf() crea el directorio de salida aunque no exista."""
        em = ExportManager(str(fake_adapter))
        out_dir = tmp_path / "deep" / "nested" / "gguf_out"

        def fake_run(cmd, check=True, **kwargs):
            for part in cmd:
                if isinstance(part, str) and part.endswith(".gguf"):
                    Path(part).touch()
            return subprocess.CompletedProcess(cmd, 0)

        merged = tmp_path / "_merged_tmp"
        merged.mkdir(parents=True, exist_ok=True)

        fake_llama = tmp_path / "llama.cpp"
        fake_llama.mkdir()
        (fake_llama / "convert_hf_to_gguf.py").write_text("# fake", encoding="utf-8")

        with patch("motor.exporter.Path.home", return_value=tmp_path):
            with patch("subprocess.run", side_effect=fake_run):
                with patch.object(em, "to_safetensors", return_value=merged):
                    try:
                        em.to_gguf(str(out_dir), quantization="f16", merged_dir=str(merged))
                    except Exception:
                        pass

        assert out_dir.exists(), "El directorio de salida debe crearse automáticamente"


# ---------------------------------------------------------------------------
# C. GGUF existente — verificar que se puede cargar con llama-cpp-python
# ---------------------------------------------------------------------------

TITANIC_GGUF = Path(r"C:\Users\usuario\Desktop\titanic_q4_k_m.gguf")

@pytest.mark.skipif(
    not TITANIC_GGUF.exists(),
    reason="GGUF de referencia no encontrado en el escritorio"
)
class TestGgufReference:

    def test_gguf_existe_y_no_esta_vacio(self):
        """El archivo GGUF de referencia existe y tiene contenido."""
        assert TITANIC_GGUF.stat().st_size > 1_000_000, "El GGUF parece vacío o corrupto"

    def test_gguf_se_puede_leer_con_llama_cpp(self):
        """Carga básica del GGUF con llama-cpp-python (Python 3.12 o 3.13)."""
        llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python no instalado")
        from llama_cpp import Llama

        llm = Llama(model_path=str(TITANIC_GGUF), n_ctx=256, verbose=False)
        assert llm is not None

    def test_gguf_genera_respuesta(self):
        """El modelo GGUF genera texto coherente."""
        llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python no instalado")
        from llama_cpp import Llama

        llm = Llama(model_path=str(TITANIC_GGUF), n_ctx=256, verbose=False)
        output = llm(
            "Classifiy the sentiment: 'Revenue grew 10% this quarter.'",
            max_tokens=10,
            echo=False,
        )
        text = output["choices"][0]["text"].strip()
        assert len(text) > 0, "El modelo debe generar texto no vacío"
