"""
tests/test_g4.py
================
G4 — score_examples() y validate(include_scores=True) en DataDigestor.

Cubre:
  A. score_examples(): estructura, rangos, criterios individuales
  B. validate(include_scores=True): integración con el reporte global
  C. Casos borde: ejemplos cortos, ruidosos, sin formato correcto
"""

from __future__ import annotations

import pytest

from motor.digestor import DataDigestor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_digestor_with(messages_list):
    """Devuelve un DataDigestor con los mensajes dados como ejemplos."""
    d = DataDigestor(task="general")
    d._examples = [
        {"messages": [
            {"role": "user",      "content": u},
            {"role": "assistant", "content": a},
        ]}
        for u, a in messages_list
    ]
    return d


GOOD_PAIR = (
    "¿Cuáles son los principales beneficios del aprendizaje automático en la industria?",
    "El aprendizaje automático permite automatizar tareas repetitivas, mejorar predicciones y optimizar procesos.",
)

NOISE_PAIR = (
    "¿Qué es eso? ☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠",
    "No sé ☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣☣",
)

SHORT_PAIR = ("OK", "Sí")

LONG_TEXT = "Texto de relleno para simular un ejemplo muy largo. " * 200  # ~10000 chars


# ===========================================================================
# A. score_examples()
# ===========================================================================

class TestScoreExamples:

    def test_devuelve_lista_del_mismo_tamano(self):
        """score_examples devuelve una lista con un item por ejemplo."""
        d = make_digestor_with([GOOD_PAIR, GOOD_PAIR, GOOD_PAIR])
        scores = d.score_examples()
        assert len(scores) == 3

    def test_estructura_por_item(self):
        """Cada item tiene index, score, length_score, noise_score, format_score, chars."""
        d = make_digestor_with([GOOD_PAIR])
        item = d.score_examples()[0]
        for key in ("index", "score", "length_score", "noise_score", "format_score", "chars"):
            assert key in item, f"Falta clave: {key}"

    def test_score_rango_0_100(self):
        """Todos los scores están entre 0 y 100."""
        d = make_digestor_with([GOOD_PAIR, NOISE_PAIR, SHORT_PAIR])
        for item in d.score_examples():
            assert 0 <= item["score"] <= 100
            assert 0 <= item["length_score"] <= 100
            assert 0 <= item["noise_score"] <= 100
            assert 0 <= item["format_score"] <= 100

    def test_indice_correlativo(self):
        """El campo 'index' coincide con la posición en la lista."""
        d = make_digestor_with([GOOD_PAIR, GOOD_PAIR, GOOD_PAIR])
        for i, item in enumerate(d.score_examples()):
            assert item["index"] == i

    def test_chars_refleja_longitud(self):
        """El campo 'chars' es la longitud total del texto del ejemplo."""
        d = DataDigestor(task="general")
        d._examples = [{"messages": [
            {"role": "user",      "content": "a" * 50},
            {"role": "assistant", "content": "b" * 50},
        ]}]
        item = d.score_examples()[0]
        # "a"*50 + " " + "b"*50 = 101 chars
        assert item["chars"] == 101

    def test_ejemplo_bueno_tiene_score_alto(self):
        """Un ejemplo bien formado y con longitud adecuada obtiene score alto (>=80)."""
        d = make_digestor_with([GOOD_PAIR])
        item = d.score_examples()[0]
        assert item["score"] >= 80

    def test_ejemplo_corto_penalizado(self):
        """Un ejemplo muy corto tiene length_score bajo."""
        d = make_digestor_with([SHORT_PAIR])
        item = d.score_examples()[0]
        assert item["length_score"] < 50

    def test_ejemplo_ruidoso_penalizado(self):
        """Un ejemplo lleno de símbolos tiene noise_score bajo."""
        d = make_digestor_with([NOISE_PAIR])
        item = d.score_examples()[0]
        assert item["noise_score"] < 70

    def test_ejemplo_largo_penalizado(self):
        """Un ejemplo con texto muy largo tiene length_score bajo."""
        d = DataDigestor(task="general")
        d._examples = [{"messages": [
            {"role": "user",      "content": LONG_TEXT},
            {"role": "assistant", "content": "Respuesta"},
        ]}]
        item = d.score_examples()[0]
        assert item["length_score"] < 80

    def test_format_score_sin_assistant(self):
        """Un mensaje sin rol 'assistant' tiene format_score < 100."""
        d = DataDigestor(task="general")
        d._examples = [{"messages": [
            {"role": "user", "content": "Pregunta sin respuesta aquí"},
        ]}]
        item = d.score_examples()[0]
        assert item["format_score"] < 100

    def test_format_score_completo(self):
        """Un mensaje con user + assistant tiene format_score == 100."""
        d = make_digestor_with([GOOD_PAIR])
        item = d.score_examples()[0]
        assert item["format_score"] == 100

    def test_dataset_vacio_devuelve_lista_vacia(self):
        """Sin ejemplos, score_examples() devuelve []."""
        d = DataDigestor(task="general")
        assert d.score_examples() == []

    def test_formato_instruccion_output(self):
        """Ejemplos en formato instruction/output también se puntúan."""
        d = DataDigestor(task="general")
        d._examples = [{
            "instruction": "¿Cuánto es 2 + 2?",
            "input": "",
            "output": "4",
        }]
        items = d.score_examples()
        assert len(items) == 1
        assert items[0]["format_score"] == 100

    def test_multiples_ejemplos_distintos_scores(self):
        """Ejemplos diferentes producen scores distintos."""
        d = make_digestor_with([GOOD_PAIR, SHORT_PAIR])
        scores = [i["score"] for i in d.score_examples()]
        assert scores[0] != scores[1]


# ===========================================================================
# B. validate(include_scores=True)
# ===========================================================================

class TestValidateIncludeScores:

    def test_include_scores_false_no_tiene_clave(self):
        """Por defecto (include_scores=False), 'quality_scores' no está en el resultado."""
        d = make_digestor_with([GOOD_PAIR] * 10)
        report = d.validate(verbose=False)
        assert "quality_scores" not in report

    def test_include_scores_true_tiene_clave(self):
        """Con include_scores=True, 'quality_scores' aparece en el resultado."""
        d = make_digestor_with([GOOD_PAIR] * 10)
        report = d.validate(verbose=False, include_scores=True)
        assert "quality_scores" in report

    def test_quality_scores_longitud_correcta(self):
        """quality_scores tiene tantos items como ejemplos."""
        n = 15
        d = make_digestor_with([GOOD_PAIR] * n)
        report = d.validate(verbose=False, include_scores=True)
        assert len(report["quality_scores"]) == n

    def test_quality_scores_consistente_con_total(self):
        """len(quality_scores) == report['total']."""
        d = make_digestor_with([GOOD_PAIR] * 8)
        report = d.validate(verbose=False, include_scores=True)
        assert len(report["quality_scores"]) == report["total"]

    def test_campos_reporte_no_modificados(self):
        """Añadir include_scores no rompe los campos habituales del reporte."""
        d = make_digestor_with([GOOD_PAIR] * 600)
        report = d.validate(verbose=False, include_scores=True)
        for key in ("total", "semaforo", "warnings", "avg_chars", "max_chars"):
            assert key in report
