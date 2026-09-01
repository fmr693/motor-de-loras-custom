"""
tests/test_server_oai.py
========================
Tests del endpoint OpenAI-compatible /v1/chat/completions (tarea A,
auditoría 10 junio 2026):

  1. Logging de interacciones en modo no-stream y stream (SSE)
  2. El `id` OpenAI (chatcmpl-...) es el interaction_id del log
     → POST /feedback funciona con él (cierra el circuito DPO)
  3. Usage real de llama-cpp (no estimación chars//4)
  4. content como lista de partes OpenAI (antes devolvía 422)
  5. Regresión: /chat/session sigue logueando con session_id/turn

No requiere modelo real: usa un stub de llama-cpp inyectado en _state.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

import motor.server as server
from motor.server import create_app, _oai_content_to_text


# ---------------------------------------------------------------------------
# Stub de llama-cpp-python
# ---------------------------------------------------------------------------

FAKE_TOOL_CALL = {
    "id": "call_abc123",
    "type": "function",
    "function": {"name": "file_organize", "arguments": '{"path": "~/Descargas"}'},
}


class FakeLlama:
    """Imita Llama.create_chat_completion en modo stream, no-stream y tools."""

    def __init__(self, reply: str = "Hola mundo"):
        self.reply = reply
        self.last_kwargs: dict = {}

    def create_chat_completion(self, messages, stream=False, **kw):
        self.last_kwargs = dict(kw, messages=messages, stream=stream)
        if kw.get("tools"):
            # Simula que el modelo decide llamar a una herramienta
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [FAKE_TOOL_CALL],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            }
        if stream:
            def gen():
                yield {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
                # Partir la respuesta en 2 chunks de contenido
                mid = len(self.reply) // 2
                yield {"choices": [{"delta": {"content": self.reply[:mid]}, "finish_reason": None}]}
                yield {"choices": [{"delta": {"content": self.reply[mid:]}, "finish_reason": None}]}
                yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            return gen()
        return {
            "choices": [{
                "message": {"role": "assistant", "content": self.reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


# ---------------------------------------------------------------------------
# Fixture: servidor en modo GGUF falso con log en tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    st = server._state
    saved = {
        "model": st.model, "llama_model": st.llama_model, "is_gguf": st.is_gguf,
        "model_path": st.model_path, "api_key": st.api_key,
        "interaction_log_path": st.interaction_log_path, "sessions": st.sessions,
    }
    st.model = None
    st.llama_model = FakeLlama()
    st.is_gguf = True
    st.model_path = "modelos/gemma-test.gguf"
    st.api_key = None
    st.interaction_log_path = str(tmp_path / "interaction_log.jsonl")
    st.sessions = {}
    try:
        yield TestClient(create_app())
    finally:
        for k, v in saved.items():
            setattr(st, k, v)


def _read_log() -> list[dict]:
    path = Path(server._state.interaction_log_path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# 1. No-stream: respuesta + usage real + log
# ---------------------------------------------------------------------------

class TestNoStream:
    def test_respuesta_y_usage_real(self, client):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "¿Quién eres?"}],
        })
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "Hola mundo"
        assert body["choices"][0]["finish_reason"] == "stop"
        # Usage real del stub, no chars//4
        assert body["usage"] == {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        }
        assert body["id"].startswith("chatcmpl-")

    def test_loguea_interaccion(self, client):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "organiza mis descargas"}],
        })
        entries = _read_log()
        assert len(entries) == 1
        e = entries[0]
        assert e["id"] == r.json()["id"]          # id OpenAI == interaction_id
        assert e["user_msg"] == "organiza mis descargas"
        assert e["assistant"] == "Hola mundo"
        assert e["feedback"] is None
        assert e["endpoint"] == "v1/chat/completions"
        assert e["model"] == "gemma-test.gguf"

    def test_user_msg_es_ultimo_mensaje_usuario(self, client):
        client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "user", "content": "primera pregunta"},
                {"role": "assistant", "content": "primera respuesta"},
                {"role": "user", "content": "segunda pregunta"},
            ],
        })
        assert _read_log()[0]["user_msg"] == "segunda pregunta"


# ---------------------------------------------------------------------------
# 2. Streaming SSE: chunks + log en finally
# ---------------------------------------------------------------------------

class TestStream:
    def test_sse_y_log(self, client):
        with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hola"}],
            "stream": True,
        }) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            raw = "".join(r.iter_text())

        datas = [l[len("data: "):] for l in raw.splitlines() if l.startswith("data: ")]
        assert datas[-1] == "[DONE]"
        chunks = [json.loads(d) for d in datas[:-1]]
        text = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        assert text == "Hola mundo"

        entries = _read_log()
        assert len(entries) == 1
        e = entries[0]
        assert e["assistant"] == "Hola mundo"
        assert e["user_msg"] == "hola"
        assert e["finish_reason"] == "stop"
        assert e["id"] == chunks[0]["id"]          # id de los chunks == log


# ---------------------------------------------------------------------------
# 3. Circuito de feedback: id OpenAI → POST /feedback → DPO-ready
# ---------------------------------------------------------------------------

class TestFeedbackRoundtrip:
    def test_feedback_con_id_openai(self, client):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "filtra mis correos"}],
        })
        chat_id = r.json()["id"]

        fb = client.post("/feedback", json={"interaction_id": chat_id, "rating": 1})
        assert fb.status_code == 200
        assert fb.json()["updated"] is True

        e = _read_log()[0]
        assert e["feedback"] == 1
        # Campos que DPOBuilder necesita para formar pares
        assert e["user_msg"] and e["assistant"]


# ---------------------------------------------------------------------------
# 4. content flexible (lista de partes / None)
# ---------------------------------------------------------------------------

class TestContentParts:
    def test_content_lista_de_partes(self, client):
        r = client.post("/v1/chat/completions", json={
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hola"},
                    {"type": "text", "text": "mundo"},
                ],
            }],
        })
        assert r.status_code == 200
        assert _read_log()[0]["user_msg"] == "hola\nmundo"

    def test_content_none_tolerado(self, client):
        r = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "assistant", "content": None},
                {"role": "user", "content": "hola"},
            ],
        })
        assert r.status_code == 200

    def test_normalizador_unitario(self):
        assert _oai_content_to_text("texto") == "texto"
        assert _oai_content_to_text(None) == ""
        assert _oai_content_to_text(
            [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}]
        ) == "a"
        assert _oai_content_to_text(["a", "b"]) == "a\nb"


# ---------------------------------------------------------------------------
# 5. Tool-calling nativo (tarea B): tools llegan a llama-cpp y
#    tool_calls vuelven al cliente
# ---------------------------------------------------------------------------

TOOLS = [{
    "type": "function",
    "function": {
        "name": "file_organize",
        "description": "Organiza archivos por tipo",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    },
}]


class TestToolCalling:
    def test_tools_se_pasan_a_llama(self, client):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "organiza mis descargas"}],
            "tools": TOOLS,
            "tool_choice": "auto",
        })
        assert r.status_code == 200
        fake = server._state.llama_model
        assert fake.last_kwargs["tools"] == TOOLS
        assert fake.last_kwargs["tool_choice"] == "auto"

    def test_respuesta_con_tool_calls(self, client):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "organiza mis descargas"}],
            "tools": TOOLS,
        })
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        assert choice["message"]["tool_calls"] == [FAKE_TOOL_CALL]
        # usage real del stub
        assert r.json()["usage"]["total_tokens"] == 28

    def test_historial_con_tool_calls_no_da_422(self, client):
        """Segunda vuelta del loop agéntico: el historial incluye la tool
        call del assistant y el resultado con role='tool'."""
        r = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "user", "content": "organiza mis descargas"},
                {"role": "assistant", "content": None, "tool_calls": [FAKE_TOOL_CALL]},
                {"role": "tool", "tool_call_id": "call_abc123",
                 "name": "file_organize", "content": "12 archivos movidos"},
            ],
        })
        assert r.status_code == 200
        # Los campos de tool llegan íntegros al modelo (el servidor inyecta
        # un system prompt en el índice 0, así que buscamos por rol)
        sent = server._state.llama_model.last_kwargs["messages"]
        asst = next(m for m in sent if m["role"] == "assistant")
        tool = next(m for m in sent if m["role"] == "tool")
        assert asst["tool_calls"] == [FAKE_TOOL_CALL]
        assert tool["tool_call_id"] == "call_abc123"
        assert tool["content"] == "12 archivos movidos"

    def test_stream_con_tools_emite_chunk_completo(self, client):
        with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "organiza mis descargas"}],
            "tools": TOOLS,
            "stream": True,
        }) as r:
            assert r.status_code == 200
            raw = "".join(r.iter_text())

        datas = [l[len("data: "):] for l in raw.splitlines() if l.startswith("data: ")]
        assert datas[-1] == "[DONE]"
        chunks = [json.loads(d) for d in datas[:-1]]
        # Primer chunk: tool_calls completas (con index); último: finish_reason
        delta = chunks[0]["choices"][0]["delta"]
        assert delta["tool_calls"][0]["function"]["name"] == "file_organize"
        assert delta["tool_calls"][0]["index"] == 0
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

        # Y la interacción queda logueada con las tool_calls
        e = _read_log()[0]
        assert e["finish_reason"] == "tool_calls"
        assert e["tool_calls"][0]["id"] == "call_abc123"

    def test_log_no_depende_de_que_el_cliente_agote_el_stream(self, client):
        """
        Un cliente agéntico que abandona el stream de tools NO puede costar la
        interacción: el camino tools es no-stream por dentro (el modelo hace
        TODO el trabajo en el primer next()), así que el dato ya existe cuando
        se emite el primer chunk. Antes, `_log_interaction` estaba DESPUÉS del
        último yield: si el cliente se iba mientras el modelo generaba,
        Starlette abandonaba el generador en el primer yield y la interacción
        se tiraba — medido contra uvicorn real: inferencia completa, 0 líneas
        de log. Y son las interacciones con herramientas las que más valen
        para el activo (Regla 11).

        Se ataca el generador directamente porque `TestClient` NO sirve: agota
        siempre la respuesta, así que un test hecho con él pasaría en verde sin
        probar nada (Regla 27). Aquí se consume UN chunk y se abandona, que es
        exactamente lo que hace un cliente que se va.
        """
        endpoint = next(r.endpoint for r in client.app.routes
                        if getattr(r, "path", None) == "/v1/chat/completions")

        class RequestFalso:            # el camino tools no lo usa; solo la firma
            async def is_disconnected(self):
                return False

        req = server._OAIRequest(
            messages=[server._OAIMessage(role="user", content="organiza mis descargas")],
            tools=TOOLS,
            stream=True,
        )
        resp = endpoint(req, RequestFalso())

        # Un solo chunk y nos vamos: el generador queda SIN agotar.
        # (Starlette envuelve el generador síncrono en uno async, de ahí el
        # __anext__ suelto en vez de un `for`.)
        import asyncio
        primero = asyncio.run(resp.body_iterator.__anext__())
        assert primero.startswith("data: ")

        entradas = _read_log()
        assert len(entradas) == 1, (
            "la interacción se perdió al abandonar el stream de tools: el "
            "modelo ya había hecho el trabajo completo"
        )
        assert entradas[0]["tool_calls"][0]["id"] == "call_abc123"

    def test_log_no_stream_incluye_tool_calls(self, client):
        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "organiza mis descargas"}],
            "tools": TOOLS,
        })
        e = _read_log()[0]
        assert e["tool_calls"] == [FAKE_TOOL_CALL]
        assert e["assistant"] == ""        # tool call pura: sin texto


# ---------------------------------------------------------------------------
# 5-bis. Clamp de temperatura consciente de familia (ficha oficial Gemma 4:
#        temperature=1.0; Qwen Q4: >0.7 incoherente)
# ---------------------------------------------------------------------------

class TestModelNullTolerado:
    def test_model_null_no_da_422(self, client):
        """El Deep Research de Odysseus sondea con model=null — el 422
        resultante mataba la investigación en segundo plano (12-jun-2026)."""
        r = client.post("/v1/chat/completions", json={
            "model": None,
            "messages": [{"role": "user", "content": "ping"}],
        })
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"]


class TestMaxTokensDefault:
    """max_tokens ausente o 0 → 2048 (el default 512 cortaba respuestas
    largas a media frase — visto en vivo con Odysseus, finish=length)."""

    def test_ausente_usa_2048(self, client):
        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hola"}],
        })
        assert server._state.llama_model.last_kwargs["max_tokens"] == 2048

    def test_cero_usa_2048(self, client):
        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hola"}],
            "max_tokens": 0,
        })
        assert server._state.llama_model.last_kwargs["max_tokens"] == 2048

    def test_explicito_se_respeta(self, client):
        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hola"}],
            "max_tokens": 100,
        })
        assert server._state.llama_model.last_kwargs["max_tokens"] == 100


class TestTemperatureClamp:
    def test_gemma_permite_temperatura_oficial_1(self, client):
        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hola"}],
            "temperature": 1.0,
        })
        # model_path del fixture es gemma-test.gguf → familia gemma → sin capar
        assert server._state.llama_model.last_kwargs["temperature"] == 1.0

    def test_familia_no_gemma_capa_a_07(self, client):
        server._state.model_path = "modelos/Qwen2.5-7B-Instruct-q4_k_m.gguf"
        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hola"}],
            "temperature": 1.0,
        })
        assert server._state.llama_model.last_kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# 5-ter. Regresión de integración: la forma EXACTA que envía Odysseus en
#        modo agente. Guarda nuestro lado pase lo que pase con Odysseus —
#        el "tema herramientas" (12-jun) fue siempre de su lado, no del nuestro.
# ---------------------------------------------------------------------------

# Multi-tool como las arma Odysseus (FUNCTION_TOOL_SCHEMAS + tool_choice)
ODYSSEUS_TOOLS = [
    {"type": "function", "function": {"name": "web_search",
        "description": "Search the web", "parameters": {"type": "object",
        "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "bash",
        "description": "Run a shell command", "parameters": {"type": "object",
        "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "note_save",
        "description": "Save a note", "parameters": {"type": "object",
        "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
        "required": ["title", "body"]}}},
]


class TestOdysseusIntegrationShape:
    """La petición que manda Odysseus en modo agente no debe romperse nunca."""

    def test_ronda1_multitool_stream_devuelve_tool_calls(self, client):
        """Ronda 1: system propio de Odysseus + user + N tools + stream + tool_choice."""
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "gemma-4-12B-it-Q4_K_M.gguf",
            "messages": [
                {"role": "system", "content": "You are Odysseus, an AI workspace agent."},
                {"role": "user", "content": "busca en internet las noticias de hoy"},
            ],
            "tools": ODYSSEUS_TOOLS,
            "tool_choice": "auto",
            "stream": True,
            "temperature": 1.0,
        }) as r:
            assert r.status_code == 200          # nunca 422
            raw = "".join(r.iter_text())
        # TODAS las tools llegaron a llama-cpp (no se perdió ninguna)
        assert len(server._state.llama_model.last_kwargs["tools"]) == 3
        # y el cliente recibe las tool_calls en formato OpenAI
        datas = [l[6:] for l in raw.splitlines() if l.startswith("data: ")]
        chunks = [json.loads(d) for d in datas[:-1]]
        assert any(c["choices"][0]["delta"].get("tool_calls") for c in chunks)

    def test_ronda2_historial_con_resultado_de_tool_no_422(self, client):
        """Ronda 2: el historial trae la tool call del assistant + el role=tool
        con el resultado. Esta forma daba 422 antes de soportar tool_call_id."""
        r = client.post("/v1/chat/completions", json={
            "model": "gemma-4-12B-it-Q4_K_M.gguf",
            "messages": [
                {"role": "system", "content": "You are Odysseus."},
                {"role": "user", "content": "busca noticias"},
                {"role": "assistant", "content": None, "tool_calls": [FAKE_TOOL_CALL]},
                {"role": "tool", "tool_call_id": "call_abc123",
                 "name": "web_search", "content": "Titular 1; Titular 2"},
            ],
            "tools": ODYSSEUS_TOOLS,
            "tool_choice": "auto",
        })
        assert r.status_code == 200
        sent = server._state.llama_model.last_kwargs["messages"]
        # el resultado de la tool llega íntegro al modelo
        tool_msg = next(m for m in sent if m["role"] == "tool")
        assert tool_msg["content"] == "Titular 1; Titular 2"
        assert tool_msg["tool_call_id"] == "call_abc123"

    def test_sin_tools_no_inventa_tool_calls(self, client):
        """Chat normal de Odysseus (sin tools) → respuesta de texto, sin tool_calls."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hola"}],
        })
        msg = r.json()["choices"][0]["message"]
        assert msg["content"]
        assert not msg.get("tool_calls")


# ---------------------------------------------------------------------------
# 6. Parser de sintaxis nativa Gemma 4 (fallback cuando llama-cpp no parsea)
# ---------------------------------------------------------------------------

class TestGemmaToolParser:
    def test_sintaxis_observada_en_vivo(self):
        """Salida real de Gemma 4 12B Q4_K_M servida por llama-cpp (10-jun-2026)."""
        from motor.server import _parse_gemma_tool_calls
        text = '<|tool_call>call:file_organize{path:<|"|>Descargas<|"|>}<tool_call|>'
        calls = _parse_gemma_tool_calls(text)
        assert calls is not None and len(calls) == 1
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "file_organize"
        assert json.loads(calls[0]["function"]["arguments"]) == {"path": "Descargas"}
        assert calls[0]["id"].startswith("call_")

    def test_texto_normal_devuelve_none(self):
        from motor.server import _parse_gemma_tool_calls
        assert _parse_gemma_tool_calls("Hola, ¿en qué puedo ayudarte?") is None
        assert _parse_gemma_tool_calls("") is None
        assert _parse_gemma_tool_calls(None) is None

    def test_rutas_windows_y_arrays(self):
        """Salida real observada en el E2E doméstico (11-jun-2026): rutas
        Windows con backslashes (escapes JSON inválidos) y argumento array."""
        from motor.server import _parse_gemma_tool_calls
        text = (
            '<|tool_call>call:file_organize{'
            'dest:<|"|>C:\\Users\\usuario\\Temp\\Imagenes<|"|>,'
            'files:[<|"|>C:\\Users\\usuario\\Temp\\a.jpg<|"|>,'
            '<|"|>C:\\Users\\usuario\\Temp\\b.jpg<|"|>]}<tool_call|>'
        )
        calls = _parse_gemma_tool_calls(text)
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["dest"] == "C:\\Users\\usuario\\Temp\\Imagenes"
        assert args["files"] == [
            "C:\\Users\\usuario\\Temp\\a.jpg",
            "C:\\Users\\usuario\\Temp\\b.jpg",
        ]

    def test_argumentos_no_parseables_se_entregan_crudos(self):
        from motor.server import _parse_gemma_tool_calls
        text = '<|tool_call>call:foo{esto no es json valido :::}<tool_call|>'
        calls = _parse_gemma_tool_calls(text)
        assert calls[0]["function"]["name"] == "foo"
        assert "raw" in json.loads(calls[0]["function"]["arguments"])

    def test_endpoint_convierte_texto_gemma_en_tool_calls(self, client):
        """Si el stub devuelve la sintaxis Gemma como texto y el cliente envió
        tools, la respuesta debe llevar tool_calls estructuradas."""
        server._state.llama_model = FakeLlama(
            reply='<|tool_call>call:file_organize{path:<|"|>Descargas<|"|>}<tool_call|>'
        )
        # FakeLlama con tools devuelve FAKE_TOOL_CALL; forzamos la rama texto
        # quitando el atajo: un stub que ignora tools y devuelve texto Gemma
        fake = server._state.llama_model

        def _no_structured(messages, stream=False, **kw):
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": fake.reply},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            }
        fake.create_chat_completion = _no_structured

        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "organiza descargas"}],
            "tools": TOOLS,
        })
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        tc = choice["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "file_organize"
        assert json.loads(tc["function"]["arguments"]) == {"path": "Descargas"}


# ---------------------------------------------------------------------------
# 6-bis. Canales <|channel>thought de Gemma 4 (fuga observada en E2E vivo)
# ---------------------------------------------------------------------------

class TestGemmaChannels:
    def test_strip_thought_vacio(self):
        from motor.server import _strip_gemma_channels
        assert _strip_gemma_channels(
            "<|channel>thought\n<channel|>Hola, ¿en qué ayudo?"
        ) == "Hola, ¿en qué ayudo?"

    def test_strip_con_razonamiento(self):
        from motor.server import _strip_gemma_channels
        assert _strip_gemma_channels(
            "<|channel>thought\nEl usuario saluda, respondo cordial.<channel|>¡Hola!"
        ) == "¡Hola!"

    def test_texto_normal_intacto(self):
        from motor.server import _strip_gemma_channels
        assert _strip_gemma_channels("Respuesta normal") == "Respuesta normal"
        assert _strip_gemma_channels("") == ""

    def test_stream_hold_retiene_marcador_parcial(self):
        from motor.server import _gemma_stream_hold
        # Chunk inicial parcial que PODRÍA ser el canal → retener
        assert _gemma_stream_hold("<|chan") == (False, "")
        # Texto que claramente no es el canal → emitir tal cual
        decided, out = _gemma_stream_hold("Hola")
        assert decided and out == "Hola"

    def test_stream_hold_suelta_tras_cierre(self):
        from motor.server import _gemma_stream_hold
        decided, out = _gemma_stream_hold(
            "<|channel>thought\npensando...<channel|>La respuesta"
        )
        assert decided and out == "La respuesta"

    def test_stream_endpoint_filtra_canal(self, client):
        """El streaming no debe emitir los tags del canal thought."""
        server._state.llama_model = FakeLlama(
            reply="<|channel>thought\n<channel|>Hola limpio"
        )
        with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hola"}],
            "stream": True,
        }) as r:
            raw = "".join(r.iter_text())
        datas = [l[len("data: "):] for l in raw.splitlines() if l.startswith("data: ")]
        chunks = [json.loads(d) for d in datas[:-1]]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text == "Hola limpio"
        assert "<|channel" not in raw
        # Y el log también queda limpio
        assert _read_log()[0]["assistant"] == "Hola limpio"


# ---------------------------------------------------------------------------
# 7. Regresión: /chat/session sigue logueando
# ---------------------------------------------------------------------------

class TestSessionRegression:
    def test_session_loguea_con_session_id_y_turn(self, client):
        r = client.post("/chat/session", json={"message": "hola sesión"})
        assert r.status_code == 200
        entries = _read_log()
        assert len(entries) == 1
        e = entries[0]
        assert e["session_id"] == r.json()["session_id"]
        assert e["turn"] == 1
        assert e["user_msg"] == "hola sesión"
        assert e["endpoint"] == "chat/session"


# ---------------------------------------------------------------------------
# 8. Robustez: desbordamiento de contexto → 400, no 500 (auditoría 14 jul 2026)
# ---------------------------------------------------------------------------

class TestContextOverflow:
    """`_raise_inference_error` clasifica el error de llama.cpp: el
    desbordamiento de contexto es culpa del prompt del cliente (400), no un
    fallo interno (500)."""

    # llama-cpp no tiene un tipo de excepción propio para esto: hay que
    # reconocerlo por TEXTO, y el texto cambia entre versiones/caminos. Ambos
    # mensajes están observados EN VIVO contra el serve real (17 jul): el
    # segundo apareció con 0.3.28 + chat handler y devolvía un 500 opaco porque
    # el fix solo conocía el primero. Parametrizado para que no vuelva a morir
    # en silencio al actualizar llama-cpp.
    @pytest.mark.parametrize("msg", [
        "Requested tokens (75470) exceed context window of 65536",
        "Prompt exceeds n_ctx: 40108 > 16384",
    ])
    def test_overflow_da_400_context_length_exceeded(self, msg):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            server._raise_inference_error(ValueError(msg))
        assert ei.value.status_code == 400
        assert "context_length_exceeded" in ei.value.detail
        assert "MOTOR_N_CTX" in ei.value.detail  # pista accionable para el operador

    def test_otros_errores_siguen_siendo_500(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            server._raise_inference_error(RuntimeError("boom"), "Error en inferencia")
        assert ei.value.status_code == 500
        assert "boom" in ei.value.detail
        assert "context_length_exceeded" not in ei.value.detail


# ---------------------------------------------------------------------------
# 6. Visión (mmproj): imágenes en /v1, degradación e higiene del log
# ---------------------------------------------------------------------------

# data-URI mínimo (el contenido no importa: el modelo está stubbeado)
_IMG_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

def _msg_con_imagen(texto="¿Qué ves?"):
    return [{"role": "user", "content": [
        {"type": "text", "text": texto},
        {"type": "image_url", "image_url": {"url": _IMG_B64}},
    ]}]


class TestHelpersVision:
    def test_has_image_part(self):
        assert server._has_image_part(_msg_con_imagen()[0]["content"])
        assert not server._has_image_part("solo texto")
        assert not server._has_image_part([{"type": "text", "text": "hola"}])

    def test_content_for_log_no_filtra_base64(self):
        # el log canónico alimenta learn/DPO: nunca debe llevar el data-URI
        out = server._content_for_log(_msg_con_imagen("mira esto")[0]["content"])
        assert "mira esto" in out and "[imagen]" in out
        assert "base64" not in out and "iVBORw0" not in out

    def test_content_for_model_sin_vision_aplana(self):
        server._state.vision_ready = False
        out = server._oai_content_for_model(_msg_con_imagen()[0]["content"])
        assert isinstance(out, str)          # aplanado (la imagen se perdería)

    def test_content_for_model_con_vision_preserva(self):
        server._state.vision_ready = True
        try:
            out = server._oai_content_for_model(_msg_con_imagen()[0]["content"])
            assert isinstance(out, list)     # se preserva para el chat handler
            assert any(p.get("type") == "image_url" for p in out)
        finally:
            server._state.vision_ready = False


class TestBuildVisionHandler:
    def test_sin_mmproj_es_modo_texto(self, monkeypatch):
        monkeypatch.delenv("MOTOR_MMPROJ", raising=False)
        assert server._build_vision_handler() is None

    def test_mmproj_inexistente_degrada_sin_romper(self, monkeypatch, capsys):
        monkeypatch.setenv("MOTOR_MMPROJ", "/no/existe/mmproj.gguf")
        assert server._build_vision_handler() is None      # no lanza
        assert "AVISO" in capsys.readouterr().out

    def test_mmproj_corrupto_degrada(self, monkeypatch, tmp_path, capsys):
        # Fichero que existe pero NO es un GGUF (descarga a medias). llama-cpp
        # no lo valida al construir el handler → sin esta comprobación se
        # anunciaba "Visión ACTIVA" y /health mentía (verificado en vivo 17-jul).
        malo = tmp_path / "mmproj.gguf"
        malo.write_bytes(b"basura no gguf")
        monkeypatch.setenv("MOTOR_MMPROJ", str(malo))
        assert server._build_vision_handler() is None      # no lanza
        assert "no es un GGUF" in capsys.readouterr().out

    def test_handler_inexistente_degrada(self, monkeypatch, tmp_path, capsys):
        fake = tmp_path / "mmproj.gguf"
        fake.write_bytes(b"GGUF")
        monkeypatch.setenv("MOTOR_MMPROJ", str(fake))
        monkeypatch.setenv("MOTOR_MMPROJ_HANDLER", "HandlerQueNoExiste")
        assert server._build_vision_handler() is None      # no lanza
        assert "AVISO" in capsys.readouterr().out


class TestVisionEndpoint:
    def test_imagen_sin_vision_da_400_explicito(self, client):
        server._state.vision_ready = False
        r = client.post("/v1/chat/completions", json={"messages": _msg_con_imagen()})
        assert r.status_code == 400        # nunca ignorarla en silencio
        assert r.json()["detail"]["type"] == "vision_not_available"

    def test_imagen_con_vision_llega_intacta_al_modelo(self, client):
        server._state.vision_ready = True
        try:
            r = client.post("/v1/chat/completions", json={"messages": _msg_con_imagen()})
            assert r.status_code == 200
            sent = server._state.llama_model.last_kwargs["messages"]
            user = next(m for m in sent if m["role"] == "user")
            assert isinstance(user["content"], list)
            assert any(p.get("type") == "image_url" for p in user["content"])
        finally:
            server._state.vision_ready = False

    def test_el_log_no_guarda_el_base64(self, client):
        server._state.vision_ready = True
        try:
            client.post("/v1/chat/completions", json={"messages": _msg_con_imagen("describe")})
            rows = _read_log()
            assert rows, "deberia haber logueado la interaccion"
            um = rows[-1]["user_msg"]
            assert "describe" in um and "[imagen]" in um
            assert "iVBORw0" not in um and "base64" not in um
        finally:
            server._state.vision_ready = False

    def test_health_expone_vision(self, client):
        server._state.vision_ready = True
        try:
            assert client.get("/health").json()["vision"] is True
        finally:
            server._state.vision_ready = False


# ---------------------------------------------------------------------------
# 7. Robustez hallada en la prueba de estrés (17 jul): imagen inválida + concurrencia
# ---------------------------------------------------------------------------

class TestImageErrors:
    """Una imagen mal formada es culpa del cliente → 400, no un 500 opaco.
    Mensajes verificados en vivo contra el chat handler de llama-cpp."""

    @pytest.mark.parametrize("msg", [
        "Incorrect padding",
        "Invalid base64-encoded string: number of data characters",
        "Failed to create bitmap from image bytes",
        "unknown url type: 'esto_no_es_una_url'",
        "<urlopen error [Errno 111] Connection refused>",
        "replace() argument 1 must be str, not dict",
    ])
    def test_imagen_invalida_da_400(self, msg):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            server._raise_inference_error(ValueError(msg))
        assert ei.value.status_code == 400
        assert ei.value.detail["type"] == "invalid_image"

    def test_log_integro_bajo_appends_y_feedback_concurrentes(self, client):
        # /feedback reescribe el log ENTERO mientras _log_interaction añade:
        # sin _LOG_LOCK se perdían interacciones (39/300 medido, ronda 4).
        # Test con la race real: 3 hilos de appends + 1 de feedbacks.
        import threading

        N, FB = 150, 15
        def appender(base):
            for i in range(N // 3):
                server._log_interaction(f"race-{base}-{i}", f"q{i}", "a", ms=1)

        def feedbacker():
            for i in range(FB):   # ids nuevos → cada POST añade 1 línea
                client.post("/feedback",
                            json={"interaction_id": f"race-fb-{i}", "rating": 1})

        ths = [threading.Thread(target=appender, args=(b,)) for b in range(3)]
        ths.append(threading.Thread(target=feedbacker))
        for t in ths: t.start()
        for t in ths: t.join()

        rows = _read_log()          # _read_log parsea: línea corrupta → excepción
        assert len(rows) == N + FB  # ni una interacción perdida

    def test_texto_mal_codificado_da_400(self):
        # un surrogate suelto en el content revienta el encode utf-8 de llama-cpp;
        # es entrada malformada del cliente → 400, no 500 (verificado en vivo).
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            server._raise_inference_error(
                ValueError("'utf-8' codec can't encode character in position 5: "
                           "surrogates not allowed"))
        assert ei.value.status_code == 400
        assert ei.value.detail["type"] == "invalid_encoding"

    def test_error_generico_sigue_500(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            server._raise_inference_error(RuntimeError("kernel panic"))
        assert ei.value.status_code == 500


class TestConcurrencySerialization:
    """llama-cpp no es thread-safe: 2 peticiones simultáneas tumbaban el proceso
    (medido en vivo). `_INFER_LOCK` debe serializar el acceso al modelo."""

    def test_no_hay_solapamiento_bajo_carga(self, tmp_path):
        import threading, time
        st = server._state
        saved = {k: getattr(st, k) for k in
                 ("model", "llama_model", "is_gguf", "model_path", "api_key",
                  "interaction_log_path", "sessions")}

        overlap = {"max": 0, "cur": 0}
        lock = threading.Lock()

        class ReentrancyFake:
            def create_chat_completion(self, messages, stream=False, **kw):
                with lock:
                    overlap["cur"] += 1
                    overlap["max"] = max(overlap["max"], overlap["cur"])
                time.sleep(0.05)                 # ventana para solaparse
                with lock:
                    overlap["cur"] -= 1
                return {"choices": [{"message": {"role": "assistant",
                        "content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                  "total_tokens": 2}}

        st.model = None
        st.llama_model = ReentrancyFake()
        st.is_gguf = True
        st.model_path = "modelos/gemma-test.gguf"
        st.api_key = None
        st.interaction_log_path = str(tmp_path / "log.jsonl")
        st.sessions = {}
        try:
            app = create_app()
            client = TestClient(app)
            errors = []

            def hit():
                try:
                    r = client.post("/v1/chat/completions",
                                    json={"messages": [{"role": "user", "content": "hola"}]})
                    assert r.status_code == 200
                except Exception as e:      # noqa
                    errors.append(e)

            ths = [threading.Thread(target=hit) for _ in range(8)]
            for t in ths: t.start()
            for t in ths: t.join()

            assert not errors, f"peticiones fallaron: {errors[:2]}"
            # LA clave: el modelo nunca se ejecutó en paralelo consigo mismo
            assert overlap["max"] == 1, f"solapamiento detectado: {overlap['max']}"
        finally:
            for k, v in saved.items():
                setattr(st, k, v)
