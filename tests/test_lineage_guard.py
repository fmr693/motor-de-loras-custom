"""
tests/test_lineage_guard.py
===========================
Tests del guard de linaje en la promoción de modelos (tarea C, auditoría
10 junio 2026).

Escenario que motiva el guard: motor-serve sirve Gemma 4, pero el ciclo CL
(continual_cycle / odysseus_bridge) está configurado con Qwen. Sin guard,
el watchdog sustituía el modelo servido por el candidato sin comprobar nada.

Cubre:
  1. _model_family() — extracción de familia desde nombre/ruta/id HF
  2. _check_promotion() — rechazo por familia distinta (flag → rejected.flag)
  3. Promoción correcta cuando la familia coincide
  4. Compatibilidad con flags antiguos (sin base_model) y formato bridge
     (campo `gguf` con la ruta del candidato)
  5. load_model() no machaca la api_key en recargas (hot-reload)
"""
from __future__ import annotations

import json

import pytest

import motor.server as server
from motor.server import _model_family, _check_promotion


# ---------------------------------------------------------------------------
# 1. _model_family
# ---------------------------------------------------------------------------

class TestModelFamily:
    @pytest.mark.parametrize("name,family", [
        ("modelos/gemma4_12b/gemma-4-12B-it-Q4_K_M.gguf", "gemma"),
        ("Qwen/Qwen2.5-7B-Instruct",                      "qwen"),
        ("modelos/domestic_7b-Q4_K_M.gguf",               None),   # no revela familia
        ("meta-llama/Llama-3.1-8B-Instruct",              "llama"),
        ("mistralai/Mistral-7B-v0.3",                     "mistral"),
        ("google/gemma-4-12b-it",                         "gemma"),
        ("",                                              None),
    ])
    def test_familias(self, name, family):
        assert _model_family(name) == family


# ---------------------------------------------------------------------------
# Fixture: estado del servidor sirviendo Gemma + recarga interceptada
# ---------------------------------------------------------------------------

@pytest.fixture()
def promo(tmp_path, monkeypatch):
    """Devuelve (flag, candidate, current, loaded) con _state sirviendo Gemma."""
    monkeypatch.setattr(server._state, "model_path",
                        "modelos/gemma4_12b/gemma-4-12B-it-Q4_K_M.gguf")
    loaded: list[str] = []
    monkeypatch.setattr(server, "load_model", lambda p, **kw: loaded.append(p))
    flag      = tmp_path / "ready.flag"
    candidate = tmp_path / "candidate.gguf"
    current   = tmp_path / "current.gguf"
    return flag, candidate, current, loaded


# ---------------------------------------------------------------------------
# 2. Rechazo por linaje
# ---------------------------------------------------------------------------

class TestRechazo:
    def test_familia_distinta_rechazada(self, promo):
        flag, candidate, current, loaded = promo
        flag.write_text(json.dumps({"base_model": "Qwen/Qwen2.5-7B-Instruct"}))
        candidate.write_text("fake gguf qwen")

        assert _check_promotion(flag, candidate, current) == "rejected"
        # No se movió el candidato, no se recargó nada
        assert candidate.exists()
        assert not current.exists()
        assert loaded == []
        # El flag queda como rejected.flag (evidencia, y rompe el bucle)
        assert not flag.exists()
        assert (flag.parent / "rejected.flag").exists()

    def test_rechazo_preserva_metadatos(self, promo):
        flag, candidate, current, _ = promo
        meta = {"base_model": "Qwen/Qwen2.5-7B-Instruct", "eval_loss": 0.1}
        flag.write_text(json.dumps(meta))
        _check_promotion(flag, candidate, current)
        saved = json.loads((flag.parent / "rejected.flag").read_text())
        assert saved == meta


# ---------------------------------------------------------------------------
# 3. Promoción correcta (misma familia)
# ---------------------------------------------------------------------------

class TestPromocion:
    def test_misma_familia_promueve(self, promo):
        flag, candidate, current, loaded = promo
        flag.write_text(json.dumps({"base_model": "google/gemma-4-12b-it"}))
        candidate.write_text("fake gguf gemma")

        assert _check_promotion(flag, candidate, current) == "promoted"
        assert not candidate.exists()          # movido a current
        assert current.read_text() == "fake gguf gemma"
        assert loaded == [str(current)]
        assert not flag.exists()               # flag consumido

    def test_sin_flag_no_hace_nada(self, promo):
        flag, candidate, current, loaded = promo
        assert _check_promotion(flag, candidate, current) is None
        assert loaded == []


# ---------------------------------------------------------------------------
# 4. Compatibilidad: flags antiguos y formato bridge
# ---------------------------------------------------------------------------

class TestCompatibilidad:
    def test_flag_sin_base_model_promueve_con_aviso(self, promo, capsys):
        flag, candidate, current, loaded = promo
        flag.write_text(json.dumps({"promoted_at": "2026-06-10T00:00:00Z"}))
        candidate.write_text("fake gguf")

        assert _check_promotion(flag, candidate, current) == "promoted"
        assert "linaje" in capsys.readouterr().out
        assert loaded == [str(current)]

    def test_flag_texto_plano_promueve(self, promo):
        flag, candidate, current, _ = promo
        flag.write_text("ready")               # formato pre-JSON
        candidate.write_text("fake gguf")
        assert _check_promotion(flag, candidate, current) == "promoted"

    def test_flag_con_ruta_gguf_propia(self, promo, tmp_path):
        """Formato del odysseus_bridge: el flag declara la ruta del GGUF
        exportado en vez de usar candidate.gguf."""
        flag, candidate, current, loaded = promo
        bridge_gguf = tmp_path / "odysseus_cl_123.gguf"
        bridge_gguf.write_text("fake gguf bridge")
        flag.write_text(json.dumps({
            "base_model": "google/gemma-4-12b-it",
            "gguf": str(bridge_gguf),
        }))

        assert _check_promotion(flag, candidate, current) == "promoted"
        assert not bridge_gguf.exists()        # movido a current
        assert current.read_text() == "fake gguf bridge"
        assert loaded == [str(current)]

    def test_modelo_servido_sin_familia_promueve(self, promo, monkeypatch):
        """Si el modelo servido no revela familia (p.ej. domestic_7b.gguf),
        no se puede verificar → comportamiento legacy (promover)."""
        flag, candidate, current, _ = promo
        monkeypatch.setattr(server._state, "model_path",
                            "modelos/domestic_7b-Q4_K_M.gguf")
        flag.write_text(json.dumps({"base_model": "Qwen/Qwen2.5-7B-Instruct"}))
        candidate.write_text("fake gguf")
        assert _check_promotion(flag, candidate, current) == "promoted"


# ---------------------------------------------------------------------------
# 5. La api_key sobrevive a recargas (hot-reload)
# ---------------------------------------------------------------------------

class TestApiKeyHotReload:
    def test_load_model_sin_api_key_no_la_borra(self, monkeypatch, tmp_path):
        st = server._state
        saved = (st.api_key, st.model_path, st.is_gguf, st.llama_model)
        try:
            st.api_key = "clave-secreta"

            class _FakeLlama:
                def __init__(self, model_path, **kw):
                    pass

            import types, sys
            fake_mod = types.ModuleType("llama_cpp")
            fake_mod.Llama = _FakeLlama
            monkeypatch.setitem(sys.modules, "llama_cpp", fake_mod)
            # detect_hardware se importa dentro de load_model
            import motor.hardware as hw
            monkeypatch.setattr(hw, "detect_hardware", lambda force=False: type(
                "HW", (), {
                    "llama_kwargs": lambda self: {},
                    "gpu_name": "X", "cuda_available": False,
                    "vram_total_gb": 0.0, "inference_profile": "test",
                    "__str__": lambda self: "hw",
                })())

            gguf = tmp_path / "m.gguf"
            gguf.write_text("x")
            server.load_model(str(gguf))           # recarga sin api_key
            assert st.api_key == "clave-secreta"   # ← antes quedaba en None

            server.load_model(str(gguf), api_key="otra")
            assert st.api_key == "otra"            # explícita sí la cambia
        finally:
            st.api_key, st.model_path, st.is_gguf, st.llama_model = saved
