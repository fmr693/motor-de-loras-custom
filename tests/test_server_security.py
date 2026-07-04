"""
tests/test_server_security.py
=============================
Tests de endurecimiento de seguridad del servidor (tarea E, auditoría
10 junio 2026):

  1. POST /agent exige API key SIEMPRE (ejecuta herramientas reales):
     - sin key configurada → 403 (deshabilitado)
     - con key configurada pero sin/mal header → 401
  2. Los endpoints de chat siguen el modelo opcional clásico:
     - sin key configurada → abiertos
     - con key configurada → Bearer obligatorio
  3. La API key sobrevive a hot-reloads (cubierto en test_lineage_guard).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import motor.server as server
from motor.server import create_app


class FakeLlama:
    def create_chat_completion(self, messages, stream=False, **kw):
        return {
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


@pytest.fixture()
def make_client(tmp_path):
    st = server._state
    saved = {
        "model": st.model, "llama_model": st.llama_model, "is_gguf": st.is_gguf,
        "model_path": st.model_path, "api_key": st.api_key,
        "interaction_log_path": st.interaction_log_path, "sessions": st.sessions,
    }

    def _make(api_key=None):
        st.model = None
        st.llama_model = FakeLlama()
        st.is_gguf = True
        st.model_path = "modelos/test.gguf"
        st.api_key = api_key
        st.interaction_log_path = str(tmp_path / "log.jsonl")
        st.sessions = {}
        return TestClient(create_app())

    try:
        yield _make
    finally:
        for k, v in saved.items():
            setattr(st, k, v)


# ---------------------------------------------------------------------------
# 1. /agent — autenticación obligatoria
# ---------------------------------------------------------------------------

class TestAgentAuth:
    def test_sin_key_configurada_403(self, make_client):
        client = make_client(api_key=None)
        r = client.post("/agent", json={"task": "organiza mis archivos"})
        assert r.status_code == 403
        assert "api-key" in r.json()["detail"].lower()

    def test_con_key_sin_header_401(self, make_client):
        client = make_client(api_key="secreta")
        r = client.post("/agent", json={"task": "hola"})
        assert r.status_code == 401

    def test_con_key_header_incorrecto_401(self, make_client):
        client = make_client(api_key="secreta")
        r = client.post("/agent", json={"task": "hola"},
                        headers={"Authorization": "Bearer incorrecta"})
        assert r.status_code == 401

    def test_con_key_correcta_pasa_auth(self, make_client):
        client = make_client(api_key="secreta")
        r = client.post("/agent", json={"task": "hola", "max_steps": 1},
                        headers={"Authorization": "Bearer secreta"})
        # La auth pasa: el resultado del agente puede ser éxito o no,
        # pero ya no es un rechazo de autenticación
        assert r.status_code not in (401, 403)


# ---------------------------------------------------------------------------
# 2. Endpoints de chat — modelo opcional clásico
# ---------------------------------------------------------------------------

class TestChatAuth:
    def test_sin_key_chat_abierto(self, make_client):
        client = make_client(api_key=None)
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hola"}],
        })
        assert r.status_code == 200

    def test_con_key_chat_exige_bearer(self, make_client):
        client = make_client(api_key="secreta")
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hola"}],
        })
        assert r.status_code == 401

        r = client.post("/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hola"}]},
                        headers={"Authorization": "Bearer secreta"})
        assert r.status_code == 200

    def test_health_siempre_abierto(self, make_client):
        client = make_client(api_key="secreta")
        assert client.get("/health").status_code == 200
