"""
tests/test_log_quality.py
=========================
Filtro de calidad del interaction_log antes de entrenar (sesión 12-jun-2026).

Diseñado contra la basura REAL observada en el log de uso vía Odysseus:
respuestas "[]"/"null"/vacías (doble-envío), truncadas (finish=length),
tool-calls sin texto, y mensajes de usuario duplicados.
"""
from __future__ import annotations

import json

import pytest

from motor.log_quality import (
    clean_text,
    is_junk,
    quality_check,
    load_quality_entries,
    load_sft_examples,
    format_report,
)


# ---------------------------------------------------------------------------
# clean_text / is_junk
# ---------------------------------------------------------------------------

class TestCleanAndJunk:
    def test_clean_quita_fences(self):
        assert clean_text("```json\n[]\n```") == "[]"
        assert clean_text("```\nhola\n```") == "hola"
        assert clean_text("  texto  ") == "texto"
        assert clean_text(None) == ""

    @pytest.mark.parametrize("txt", ["", "[]", "null", "{}", "  ", "```json\n[]\n```", "N/A", "..."])
    def test_junk_detectado(self, txt):
        assert is_junk(txt) is True

    @pytest.mark.parametrize("txt", ["Hola, ¿en qué ayudo?", "[1, 2, 3]", "La respuesta es 42."])
    def test_contenido_real_no_es_junk(self, txt):
        assert is_junk(txt) is False


# ---------------------------------------------------------------------------
# quality_check — veredicto por entrada
# ---------------------------------------------------------------------------

class TestQualityCheck:
    def _e(self, **kw):
        base = {"user_msg": "una pregunta razonable", "assistant": "una respuesta suficientemente larga"}
        base.update(kw)
        return base

    def test_entrada_buena(self):
        assert quality_check(self._e()) == (True, "ok")

    def test_user_vacio(self):
        assert quality_check(self._e(user_msg=""))[1] == "user_vacio"

    def test_assistant_basura(self):
        assert quality_check(self._e(assistant="[]"))[1] == "assistant_basura"
        assert quality_check(self._e(assistant="```json\n[]\n```"))[1] == "assistant_basura"
        assert quality_check(self._e(assistant=None))[1] == "assistant_basura"

    def test_assistant_corto(self):
        assert quality_check(self._e(assistant="ok"))[1] == "assistant_corto"

    def test_truncado(self):
        assert quality_check(self._e(finish_reason="length"))[1] == "truncado"

    @pytest.mark.parametrize("valor", [123, ["lista"], {"d": 1}, 3.14])
    def test_user_msg_no_string_no_lanza(self, valor):
        # un log real acumula user_msg no-string por clientes que mandan basura;
        # el contrato es "Nunca lanza" (si no, tumba learn --auto / DPO). Ronda 5.
        ok, motivo = quality_check(self._e(user_msg=valor))   # no debe lanzar
        assert isinstance(ok, bool) and isinstance(motivo, str)
        # se puede desactivar
        assert quality_check(self._e(finish_reason="length"), reject_truncated=False)[0] is True

    def test_tool_call(self):
        assert quality_check(self._e(assistant="", finish_reason="tool_calls"))[1] == "tool_call"
        assert quality_check(self._e(tool_calls=[{"id": "x"}]))[1] == "tool_call"

    def test_no_dict(self):
        assert quality_check("no soy dict")[0] is False


# ---------------------------------------------------------------------------
# load_quality_entries — sobre un log realista
# ---------------------------------------------------------------------------

# Réplica de los patrones del log real (12-jun-2026)
REAL_PATTERNS = [
    {"user_msg": "hola, preséntate por favor", "assistant": "¡Hola! Soy un asistente de IA local.", "feedback": None, "finish_reason": "stop"},
    {"user_msg": "termina de responder", "assistant": "[]", "feedback": None, "finish_reason": "stop"},                       # basura
    {"user_msg": "consulta internet", "assistant": "```json\n[]\n```", "feedback": None, "finish_reason": "stop"},            # basura con fence
    {"user_msg": "busca en internet", "assistant": "", "feedback": None, "finish_reason": "tool_calls", "tool_calls": [{"id": "c1"}]},  # tool call
    {"user_msg": "explica X", "assistant": "Para explicar X necesito primero definir el contexto en el que", "feedback": None, "finish_reason": "length"},  # truncado
    {"user_msg": "hola, preséntate por favor", "assistant": "¡Hola! Soy un asistente de IA local.", "feedback": None, "finish_reason": "stop"},  # duplicado del 1º
    {"user_msg": "conversa", "assistant": "null", "feedback": None, "finish_reason": "stop"},                                 # basura
    {"user_msg": "dime un dato curioso", "assistant": "El pulpo tiene tres corazones y sangre azul.", "feedback": 1, "finish_reason": "stop"},
]


@pytest.fixture()
def real_log(tmp_path):
    p = tmp_path / "interaction_log.jsonl"
    p.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in REAL_PATTERNS) + "\n",
                 encoding="utf-8")
    return p


class TestLoadQualityEntries:
    def test_filtra_la_basura_real(self, real_log):
        report: dict = {}
        kept = load_quality_entries(real_log, report=report)
        # De 8 entradas: 2 buenas únicas (presentación + dato curioso).
        assert len(kept) == 2
        assert report["assistant_basura"] == 3   # "[]", "```json[]```", "null"
        assert report["tool_call"] == 1
        assert report["truncado"] == 1
        assert report["duplicado"] == 1

    def test_dedup_desactivable(self, real_log):
        kept = load_quality_entries(real_log, dedup=False)
        # Sin dedup, la presentación duplicada cuenta dos veces → 3 buenas
        assert len(kept) == 3

    def test_filtro_por_feedback(self, real_log):
        kept = load_quality_entries(real_log, feedback={1})
        assert len(kept) == 1
        assert kept[0]["assistant"].startswith("El pulpo")

    def test_dpo_no_descarta_disliked_coherente(self, tmp_path):
        """Una respuesta mala pero COHERENTE con 👎 debe pasar (es el rejected)."""
        p = tmp_path / "log.jsonl"
        p.write_text(json.dumps({
            "user_msg": "¿cuánto es 2+2?",
            "assistant": "Creo que son cinco, no estoy seguro la verdad.",
            "feedback": -1, "finish_reason": "stop",
        }) + "\n", encoding="utf-8")
        kept = load_quality_entries(p, feedback={1, -1}, dedup=False)
        assert len(kept) == 1   # coherente aunque sea incorrecta → se conserva


class TestLoadSftExamples:
    def test_formato_chat(self, real_log):
        ex = load_sft_examples(real_log)
        assert len(ex) == 2
        assert ex[0]["messages"][0]["role"] == "user"
        assert ex[0]["messages"][1]["role"] == "assistant"

    def test_excluye_feedback_negativo(self, tmp_path):
        p = tmp_path / "log.jsonl"
        p.write_text(json.dumps({
            "user_msg": "una pregunta cualquiera larga",
            "assistant": "una respuesta coherente y suficientemente larga",
            "feedback": -1, "finish_reason": "stop",
        }) + "\n", encoding="utf-8")
        # include_unrated acepta None y 1, NUNCA -1
        assert load_sft_examples(p) == []


class TestReport:
    def test_format_report(self):
        r = {"ok": 5, "assistant_basura": 2, "truncado": 1}
        out = format_report(r)
        assert "5/8" in out
        assert "assistant_basura=2" in out
