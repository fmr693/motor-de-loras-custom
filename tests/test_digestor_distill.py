"""
tests/test_digestor_distill.py
==============================
Reconversión del Digestor (jul 2026): modos explícitos (F1) + destilación de
transcripciones markdown con higiene (F2).

Todo determinista (sin LLM): valida el flujo standalone, que es el requisito
duro. Principio: calidad del dato de salida > cantidad.
"""
from __future__ import annotations

import pytest

from motor.digestor import (
    DataDigestor,
    _VALID_MODES,
    _md_role_match,
    _parse_md_dialogue,
    _strip_identity,
    _is_refusal,
)


# ---------------------------------------------------------------------------
# F1 — Modos explícitos + task opcional
# ---------------------------------------------------------------------------

class TestModos:
    def test_modos_validos(self):
        assert {"classify", "distill", "knowledge", "vlm"} <= _VALID_MODES

    def test_task_opcional_en_distill(self):
        d = DataDigestor(mode="distill", auto_enrich=False)
        assert d.task == ""
        assert d.mode == "distill"
        # system prompt del modo, no el de classify
        assert "experto" in d.system_prompt.lower()

    def test_classify_retrocompat(self):
        d = DataDigestor(task="¿Abusivo? SÍ/NO", auto_enrich=False)
        assert d.mode == "classify"
        assert d.task == "¿Abusivo? SÍ/NO"

    def test_modo_invalido_falla(self):
        with pytest.raises(ValueError):
            DataDigestor(mode="inventado", auto_enrich=False)


# ---------------------------------------------------------------------------
# Detección de marcadores de rol
# ---------------------------------------------------------------------------

class TestRoleMatch:
    @pytest.mark.parametrize("line,role", [
        ("## Human", "user"),
        ("## Assistant", "assistant"),
        ("### Human", "user"),
        ("**User:**", "user"),
        ("**Assistant:**", "assistant"),
        ("User:", "user"),
        ("Assistant:", "assistant"),
        ("You:", "user"),
        ("Claude:", "assistant"),
        ("ChatGPT:", "assistant"),
        ("Gemini:", "assistant"),
    ])
    def test_marcadores_reconocidos(self, line, role):
        m = _md_role_match(line)
        assert m is not None and m[0] == role

    def test_contenido_inline(self):
        assert _md_role_match("User: hola qué tal") == ("user", "hola qué tal")

    def test_frase_que_empieza_por_rol_no_es_marcador(self):
        # sin ':' y con texto después → contenido, no marcador
        assert _md_role_match("You should try this") is None

    def test_linea_de_contenido_normal(self):
        assert _md_role_match("El coste es 5 euros") is None


# ---------------------------------------------------------------------------
# Higiene de destilación
# ---------------------------------------------------------------------------

class TestHigiene:
    def test_strip_identity_en_preserva_contenido(self):
        t = "As an AI language model, I cannot feel. Here is the answer: 42."
        out = _strip_identity(t)
        assert "As an AI" not in out
        assert "42" in out                      # el contenido útil sobrevive

    def test_strip_identity_es_preserva_contenido(self):
        t = "Soy Claude, un asistente de IA. La respuesta es 4."
        out = _strip_identity(t)
        assert "Soy Claude" not in out
        assert "La respuesta es 4." in out

    def test_refusal_corto_detectado(self):
        assert _is_refusal("I'm sorry, but I can't help with that.")
        assert _is_refusal("Lo siento, no puedo ayudarte con eso.")

    def test_respuesta_larga_util_no_es_refusal(self):
        t = "You can't divide by zero because it is undefined. " + "detalle " * 80
        assert not _is_refusal(t)


# ---------------------------------------------------------------------------
# Parseo de transcripción
# ---------------------------------------------------------------------------

class TestParseDialogue:
    def test_fusiona_turnos_consecutivos(self):
        text = "## Human\nhola\n\n## Human\nqué tal\n\n## Assistant\nbien"
        turns = _parse_md_dialogue(text)
        assert len(turns) == 2
        assert turns[0][0] == "user"
        assert "hola" in turns[0][1] and "qué tal" in turns[0][1]

    def test_ignora_preambulo(self):
        text = "Exportado el 2026\n\n## Human\nhola\n## Assistant\nadiós"
        turns = _parse_md_dialogue(text)
        assert turns[0] == ["user", "hola"]


# ---------------------------------------------------------------------------
# F2 — from_markdown_dialogue E2E
# ---------------------------------------------------------------------------

class TestFromMarkdownDialogue:
    def _write(self, tmp_path, name, content):
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_e2e_construye_pares(self, tmp_path):
        md = ("## Human\n¿Qué es Python?\n## Assistant\nUn lenguaje de "
              "programación.\n## Human\n¿Y Rust?\n## Assistant\nUn lenguaje "
              "de sistemas.")
        f = self._write(tmp_path, "chat.md", md)
        d = DataDigestor(mode="distill", auto_enrich=False)
        d.from_markdown_dialogue(f)
        ex = d.get_examples()
        assert len(ex) == 1
        msgs = ex[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert len(msgs) == 5                    # system + 2 pares
        assert msgs[1]["role"] == "user" and "Python" in msgs[1]["content"]
        assert msgs[2]["role"] == "assistant"

    def test_descarta_par_de_rechazo(self, tmp_path):
        md = ("User: haz algo dudoso\nAssistant: Lo siento, no puedo ayudarte "
              "con eso.\nUser: ¿Cuánto es 2+2?\nAssistant: Son 4.")
        f = self._write(tmp_path, "chat.md", md)
        d = DataDigestor(mode="distill", auto_enrich=False)
        d.from_markdown_dialogue(f)
        msgs = d.get_examples()[0]["messages"]
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert any("4" in c for c in asst)
        assert not any("no puedo" in c.lower() for c in asst)

    def test_strip_identity_en_salida(self, tmp_path):
        md = ("User: hola\nAssistant: Soy Claude, un asistente de IA. "
              "La respuesta correcta es sí.")
        f = self._write(tmp_path, "chat.md", md)
        d = DataDigestor(mode="distill", auto_enrich=False)
        d.from_markdown_dialogue(f)
        a = [m["content"] for m in d.get_examples()[0]["messages"]
             if m["role"] == "assistant"][0]
        assert "Soy Claude" not in a
        assert "sí" in a

    def test_carpeta_recursiva(self, tmp_path):
        self._write(tmp_path, "a.md", "## Human\nhola\n## Assistant\nadiós amigo")
        self._write(tmp_path, "b.md", "## Human\nqué tal\n## Assistant\nbien gracias")
        d = DataDigestor(mode="distill", auto_enrich=False)
        d.from_markdown_dialogue(tmp_path)
        assert len(d.get_examples()) == 2

    def test_archivo_sin_dialogo_se_omite(self, tmp_path):
        f = self._write(tmp_path, "notas.md", "Solo texto plano sin roles.\nOtra línea.")
        d = DataDigestor(mode="distill", auto_enrich=False)
        d.from_markdown_dialogue(f)
        assert d.get_examples() == []


# ---------------------------------------------------------------------------
# F3 — Modo conocimiento (documento → dataset)
# ---------------------------------------------------------------------------

import motor.digestor as _dig
from motor.digestor import _chunk_text, _chunk_topic


class TestChunking:
    def test_trocea_por_parrafos(self):
        text = "Párrafo uno.\n\nPárrafo dos.\n\nPárrafo tres."
        chunks = _chunk_text(text, chunk_chars=20)
        assert len(chunks) >= 2
        assert all(c.strip() for c in chunks)

    def test_tema_desde_encabezado(self):
        assert _chunk_topic("# Cláusula de rescisión\nTexto...") == "Cláusula de rescisión"


class TestKnowledge:
    def _doc(self, tmp_path, text, name="doc.md"):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return p

    def test_nivel_completion(self, tmp_path):
        f = self._doc(tmp_path, "Un texto de dominio.\n\nOtro párrafo distinto aquí.")
        d = DataDigestor(mode="knowledge", auto_enrich=False)
        d.from_document_knowledge(f, level="completion", chunk_chars=25)
        ex = d.get_examples()
        assert ex and all("text" in e for e in ex)   # formato continued-pretraining

    def test_nivel_template_es_chatml(self, tmp_path):
        f = self._doc(tmp_path, "# Rescisión\nEl contrato puede rescindirse con 30 días.")
        d = DataDigestor(mode="knowledge", auto_enrich=False)
        d.from_document_knowledge(f, level="template")
        ex = d.get_examples()
        assert ex
        msgs = ex[0]["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert "Rescisión" in msgs[1]["content"]           # pregunta usa el tema
        assert "30 días" in msgs[2]["content"]             # respuesta = contenido

    def test_auto_degrada_a_template_sin_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_dig, "_llm_reachable", lambda *a, **k: False)
        f = self._doc(tmp_path, "Contenido de dominio para digerir aquí.")
        d = DataDigestor(mode="knowledge", auto_enrich=False)
        d.from_document_knowledge(f, level="auto")
        # standalone: produjo ejemplos ChatML por plantilla, sin tocar red
        assert d.get_examples()
        assert "messages" in d.get_examples()[0]

    def test_llm_pedido_offline_degrada(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_dig, "_llm_reachable", lambda *a, **k: False)
        called = []
        monkeypatch.setattr(_dig, "_llm_chat", lambda *a, **k: called.append(1) or "{}")
        f = self._doc(tmp_path, "Texto que digerir sin endpoint disponible.")
        d = DataDigestor(mode="knowledge", auto_enrich=False)
        d.from_document_knowledge(f, level="llm")
        assert called == []                     # NO se llamó al LLM (offline)
        assert d.get_examples()                 # pero SÍ produjo dataset (plantilla)

    def test_nivel_llm_con_mock(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_dig, "_llm_reachable", lambda *a, **k: True)
        monkeypatch.setattr(
            _dig, "_llm_chat",
            lambda *a, **k: '{"pairs":[{"q":"¿Qué plazo hay?","a":"30 días."}]}',
        )
        f = self._doc(tmp_path, "El contrato se rescinde con 30 días de aviso.")
        d = DataDigestor(mode="knowledge", auto_enrich=False)
        d.from_document_knowledge(f, level="auto")
        msgs = d.get_examples()[0]["messages"]
        assert msgs[1]["content"] == "¿Qué plazo hay?"
        assert msgs[2]["content"] == "30 días."

    def test_nivel_invalido_falla(self, tmp_path):
        f = self._doc(tmp_path, "texto")
        d = DataDigestor(mode="knowledge", auto_enrich=False)
        with pytest.raises(ValueError):
            d.from_document_knowledge(f, level="inventado")
