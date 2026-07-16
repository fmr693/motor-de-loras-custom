"""
tests/test_reflection.py
========================
Pase de reflexión (feedback implícito por LLM-juez, aprendizaje híbrido).

No requiere modelo real: se mockea `motor.reflection._chat` para devolver
JSON canónico y comprobar la lógica de agrupación, troceado, parseo tolerante,
filtro de confianza, acarreo de id real y extracción de pares de corrección.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import motor.reflection as refl
from motor.reflection import (
    ReflectionJudge,
    _chunk,
    _extract_json,
    _load_sessions,
    format_report,
)


def _write_log(tmp_path: Path, rows: list) -> Path:
    p = tmp_path / "log.jsonl"
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# 1. Troceado de sesiones largas
# ---------------------------------------------------------------------------

class TestChunk:
    def test_sin_trocear_si_cabe(self):
        assert _chunk([1, 2, 3], size=10, overlap=1) == [[1, 2, 3]]

    def test_ventanas_solapadas(self):
        chunks = _chunk(list(range(1, 6)), size=3, overlap=1)  # step=2
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [3, 4, 5]
        # el último elemento de una ventana solapa con el primero de la siguiente
        assert chunks[0][-1] == chunks[1][0]
        # cubre hasta el final
        assert chunks[-1][-1] == 5


# ---------------------------------------------------------------------------
# 2. Parseo tolerante del JSON del juez
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_json_plano(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_json_con_fences(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_embebido(self):
        assert _extract_json('texto antes {"a": 1} texto después') == {"a": 1}

    def test_basura_devuelve_none(self):
        assert _extract_json("esto no es json") is None


# ---------------------------------------------------------------------------
# 3. Agrupación en conversaciones
# ---------------------------------------------------------------------------

class TestLoadSessions:
    def test_sesiones_reales_y_sueltas(self, tmp_path):
        rows = [
            {"id": "s1_t1", "session_id": "s1", "turn": 1, "user_msg": "a", "assistant": "A"},
            {"id": "s1_t2", "session_id": "s1", "turn": 2, "user_msg": "b", "assistant": "B"},
            {"id": "c1", "session_id": None, "turn": None, "user_msg": "c", "assistant": "C"},
            {"id": "c2", "session_id": None, "turn": None, "user_msg": "d", "assistant": "D"},
            # sin assistant → ruido, se ignora
            {"id": "n", "session_id": "s3", "turn": 1, "user_msg": "x", "assistant": ""},
        ]
        s = _load_sessions(_write_log(tmp_path, rows))
        assert set(s.keys()) == {"s1", "solo:c1", "solo:c2"}
        assert len(s["s1"]) == 2
        assert len(s["solo:c1"]) == 1

    def test_orden_por_turno_tolera_none(self, tmp_path):
        rows = [
            {"id": "s1_t2", "session_id": "s1", "turn": 2, "user_msg": "b", "assistant": "B"},
            {"id": "s1_t1", "session_id": "s1", "turn": 1, "user_msg": "a", "assistant": "A"},
        ]
        s = _load_sessions(_write_log(tmp_path, rows))
        assert [e["turn"] for e in s["s1"]] == [1, 2]


# ---------------------------------------------------------------------------
# 4. run(): veredictos, filtro de confianza, id real, pares
# ---------------------------------------------------------------------------

class TestRun:
    def _sesion_2_turnos(self, tmp_path):
        rows = [
            {"id": "s1_t1", "session_id": "s1", "turn": 1,
             "user_msg": "pregunta", "assistant": "resp mala"},
            {"id": "s1_t2", "session_id": "s1", "turn": 2,
             "user_msg": "no, corrige", "assistant": "resp buena"},
        ]
        return _write_log(tmp_path, rows)

    def test_veredictos_y_pares(self, tmp_path, monkeypatch):
        canned = json.dumps({
            "verdicts": [
                {"turn": 1, "label": "error", "confidence": 0.9, "reason": "corrige"},
                # confianza < min → se filtra
                {"turn": 2, "label": "acierto", "confidence": 0.5, "reason": "ok"},
            ],
            "correction_pairs": [
                {"turn": 1, "rejected": "resp mala", "chosen": "resp buena"},
            ],
        })
        monkeypatch.setattr(refl, "_chat", lambda *a, **k: canned)
        res = ReflectionJudge(self._sesion_2_turnos(tmp_path), min_confidence=0.6).run()

        assert res.sessions_judged == 1
        assert len(res.labels) == 1                # solo el turno 1 pasa el filtro
        v = res.labels[0]
        assert v.label == -1 and v.turn == 1
        assert v.id == "s1_t1"                      # id REAL acarreado (join en DPO)
        assert v.source == "reflection"

        assert len(res.pairs) == 1
        p = res.pairs[0]
        assert p["prompt"] == "pregunta"           # prompt tomado del log por turno
        assert p["chosen"] == "resp buena"
        assert p["rejected"] == "resp mala"
        assert p["source"] == "reflection"

    def test_turno_unico_se_salta(self, tmp_path, monkeypatch):
        rows = [{"id": "c1", "session_id": None, "turn": None,
                 "user_msg": "x", "assistant": "Y"}]
        log = _write_log(tmp_path, rows)
        called = []
        monkeypatch.setattr(refl, "_chat", lambda *a, **k: called.append(1) or "{}")
        res = ReflectionJudge(log).run()
        assert called == []                        # el juez NO se llama para 1 turno
        assert res.sessions_judged == 0

    def test_salida_malformada_no_es_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(refl, "_chat", lambda *a, **k: "esto no es json")
        res = ReflectionJudge(self._sesion_2_turnos(tmp_path)).run()
        assert res.sessions_judged == 0
        assert res.sessions_failed == 1

    def test_write_y_report(self, tmp_path, monkeypatch):
        canned = json.dumps({
            "verdicts": [{"turn": 1, "label": "acierto", "confidence": 0.8, "reason": "ok"}],
            "correction_pairs": [],
        })
        monkeypatch.setattr(refl, "_chat", lambda *a, **k: canned)
        j = ReflectionJudge(self._sesion_2_turnos(tmp_path))
        res = j.run()
        out = j.write(res, tmp_path / "out")
        assert out["labels"].exists() and out["pairs"].exists()
        recs = [json.loads(l) for l in out["labels"].read_text(encoding="utf-8").splitlines()]
        assert len(recs) == 1 and recs[0]["label"] == 1
        assert "1 aciertos" in format_report(res)


# ---------------------------------------------------------------------------
# 5. Integración con DPOBuilder (feedback implícito + prioridad humana)
# ---------------------------------------------------------------------------

class TestDPOIntegration:
    def test_reflection_se_fusiona_en_dpo(self, tmp_path):
        from motor.dpo_trainer import DPOBuilder

        # Log con un mismo prompt respondido dos veces (una buena, una mala),
        # SIN feedback humano — la reflexión aportará las etiquetas.
        rows = [
            {"id": "a", "session_id": "s", "turn": 1, "user_msg": "¿capital?",
             "assistant": "Madrid", "feedback": None},
            {"id": "b", "session_id": "s", "turn": 2, "user_msg": "¿capital?",
             "assistant": "Lisboa", "feedback": None},
        ]
        log = _write_log(tmp_path, rows)

        refl_dir = tmp_path / "refl"
        refl_dir.mkdir()
        (refl_dir / "reflection_labels.jsonl").write_text(
            json.dumps({"id": "a", "label": 1, "source": "reflection"}) + "\n" +
            json.dumps({"id": "b", "label": -1, "source": "reflection"}) + "\n",
            encoding="utf-8",
        )
        (refl_dir / "reflection_pairs.jsonl").write_text("", encoding="utf-8")

        builder = DPOBuilder(log, min_pairs=1, reflection_dir=str(refl_dir))
        pairs = builder.build_pairs()
        # mismo prompt con +1 (Madrid) y -1 (Lisboa) → un par de preferencia
        assert len(pairs) == 1
        assert pairs[0]["chosen"] == "Madrid"
        assert pairs[0]["rejected"] == "Lisboa"

    def test_feedback_humano_tiene_prioridad(self, tmp_path):
        from motor.dpo_trainer import DPOBuilder

        # 'a' YA tiene feedback humano +1: la etiqueta de reflexión sobre 'a'
        # (que dice -1) debe IGNORARSE.
        rows = [
            {"id": "a", "session_id": "s", "turn": 1, "user_msg": "¿capital?",
             "assistant": "Madrid", "feedback": 1},
            {"id": "b", "session_id": "s", "turn": 2, "user_msg": "¿capital?",
             "assistant": "Lisboa", "feedback": -1},
        ]
        log = _write_log(tmp_path, rows)
        refl_dir = tmp_path / "refl"
        refl_dir.mkdir()
        (refl_dir / "reflection_labels.jsonl").write_text(
            json.dumps({"id": "a", "label": -1, "source": "reflection"}) + "\n",
            encoding="utf-8",
        )
        (refl_dir / "reflection_pairs.jsonl").write_text("", encoding="utf-8")

        builder = DPOBuilder(log, min_pairs=1, reflection_dir=str(refl_dir))
        labels, _ = builder._load_reflection({"a"})   # 'a' es explícito
        assert labels == []                            # se descarta la etiqueta inferida
