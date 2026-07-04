"""
tests/test_s6.py
================
Sprint 6.1 — generate_tool_calls() en DataDigestor.

Cubre:
  A. Generación sintética (n_per_tool)
  B. Ejemplos explícitos (str y dict)
  C. Formato react y function_call
  D. Sistema de herramientas en el prompt
  E. Auxiliares: _match_tool, _sample_args, _generate_tool_requests
  F. Integración: encadenado con to_jsonl()

No requiere GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motor.digestor import DataDigestor

# ---------------------------------------------------------------------------
# Herramientas de prueba
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name":        "note_save",
        "description": "Guarda una nota con título y cuerpo",
        "parameters": {
            "title":    {"type": "str"},
            "body":     {"type": "str"},
            "notebook": {"type": "str"},
        },
    },
    {
        "name":        "file_organize",
        "description": "Organiza archivos en una carpeta de destino",
        "parameters": {
            "dest": {"type": "str"},
            "files": {"type": "list"},
        },
    },
    {
        "name":        "search_files",
        "description": "Busca archivos que contienen una consulta",
        "parameters": {
            "query": {"type": "str"},
            "path":  {"type": "str"},
        },
    },
]

TOOLS_MINIMAL = [
    {"name": "ping", "description": "Hace ping a un host"},
]

# ===========================================================================
# A. Generación sintética
# ===========================================================================

class TestGeneracionSintetica:

    def test_genera_n_por_herramienta(self):
        """Genera exactamente n_per_tool ejemplos por herramienta."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=4)
        assert len(d._examples) == len(TOOLS) * 4

    def test_genera_n_por_herramienta_default(self):
        """n_per_tool=5 por defecto."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS)
        assert len(d._examples) == len(TOOLS) * 5

    def test_una_herramienta(self):
        """Con una sola herramienta genera n_per_tool ejemplos."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS_MINIMAL, n_per_tool=3)
        assert len(d._examples) == 3

    def test_herramienta_sin_templates_conocidos(self):
        """Herramienta sin templates en _TOOL_REQUEST_TEMPLATES usa fallback."""
        tools = [{"name": "custom_tool", "description": "Hace algo especial",
                  "parameters": {"x": {"type": "str"}}}]
        d = DataDigestor(task="agente")
        d.generate_tool_calls(tools, n_per_tool=2)
        assert len(d._examples) == 2

    def test_encadenado(self):
        """Dos llamadas a generate_tool_calls() acumulan ejemplos."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS[:1], n_per_tool=3)
        d.generate_tool_calls(TOOLS[1:2], n_per_tool=2)
        assert len(d._examples) == 5

    def test_seed_reproducible(self):
        """Mismo seed → mismos ejemplos."""
        d1 = DataDigestor(task="agente")
        d1.generate_tool_calls(TOOLS, n_per_tool=3, seed=7)
        d2 = DataDigestor(task="agente")
        d2.generate_tool_calls(TOOLS, n_per_tool=3, seed=7)
        assert [ex["messages"][-1]["content"] for ex in d1._examples] == \
               [ex["messages"][-1]["content"] for ex in d2._examples]


# ===========================================================================
# B. Ejemplos explícitos
# ===========================================================================

class TestEjemplosExplicitos:

    def test_explicito_string_cuenta(self):
        """Un ejemplo explícito de tipo str añade 1 ejemplo extra."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=0, examples=["guarda una nota sobre el proyecto"])
        assert len(d._examples) == 1

    def test_explicito_dict_completo(self):
        """Un ejemplo explícito dict completo se usa tal cual."""
        ex = {
            "user": "busca los informes del Q1",
            "tool": "search_files",
            "args": {"query": "Q1", "path": "~/Documentos"},
        }
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=0, examples=[ex])
        assert len(d._examples) == 1
        # El mensaje del usuario debe contener el texto del ejemplo
        users = [e["messages"][1]["content"] for e in d._examples]
        assert ex["user"] in users

    def test_explicitos_mas_sinteticos(self):
        """Explícitos + sintéticos se suman correctamente."""
        exs = ["guarda una nota", "busca archivos del proyecto"]
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=2, examples=exs)
        # 2 explícitos + 3 tools × 2 = 8
        assert len(d._examples) == 2 + len(TOOLS) * 2

    def test_sin_ejemplos_explicitos(self):
        """examples=None no rompe nada."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=1, examples=None)
        assert len(d._examples) == len(TOOLS)


# ===========================================================================
# C. Formato react
# ===========================================================================

class TestFormatoReact:

    def test_react_tiene_thought_y_final(self):
        """Formato react: el assistant siempre tiene Thought: y Final Answer:."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=2, format="react")
        for ex in d._examples:
            asst = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            assert "Thought:" in asst
            assert "Action:" in asst
            assert "Action Input:" in asst
            assert "Observation:" in asst
            assert "Final Answer:" in asst

    def test_react_action_input_es_json_valido(self):
        """Action Input: debe ser JSON parseable."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=3, format="react")
        for ex in d._examples:
            asst = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            # Extraer la línea "Action Input: {...}"
            for line in asst.splitlines():
                if line.startswith("Action Input:"):
                    json_part = line[len("Action Input:"):].strip()
                    parsed = json.loads(json_part)
                    assert isinstance(parsed, dict)
                    break

    def test_react_action_menciona_herramienta(self):
        """La línea Action: debe contener el nombre de una herramienta válida."""
        valid_names = {t["name"] for t in TOOLS}
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=2, format="react")
        for ex in d._examples:
            asst = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            for line in asst.splitlines():
                if line.startswith("Action:"):
                    tool_name = line[len("Action:"):].strip()
                    assert tool_name in valid_names
                    break


# ===========================================================================
# D. Formato function_call
# ===========================================================================

class TestFormatoFunctionCall:

    def test_fc_assistant_es_json(self):
        """Formato function_call: el assistant es JSON con 'tool' y 'args'."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=3, format="function_call")
        for ex in d._examples:
            asst = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            parsed = json.loads(asst)
            assert "tool" in parsed
            assert "args" in parsed

    def test_fc_tool_es_valido(self):
        """El campo 'tool' del JSON debe ser una herramienta de la lista."""
        valid_names = {t["name"] for t in TOOLS}
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=3, format="function_call")
        for ex in d._examples:
            asst = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            parsed = json.loads(asst)
            assert parsed["tool"] in valid_names


# ===========================================================================
# E. Sistema describe herramientas disponibles
# ===========================================================================

class TestSystemPrompt:

    def test_system_contiene_nombres_herramientas(self):
        """El system prompt debe listar los nombres de las herramientas."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=1)
        for ex in d._examples:
            system = next(m["content"] for m in ex["messages"] if m["role"] == "system")
            for tool in TOOLS:
                assert tool["name"] in system

    def test_system_contiene_herramientas_disponibles(self):
        """El system prompt debe mencionar las herramientas de alguna forma."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS_MINIMAL, n_per_tool=1)
        system = next(
            m["content"] for m in d._examples[0]["messages"] if m["role"] == "system"
        )
        assert "ping" in system


# ===========================================================================
# F. Auxiliares
# ===========================================================================

class TestAuxiliares:

    def test_match_tool_retorna_herramienta(self):
        """_match_tool devuelve una herramienta (dict) o None."""
        d = DataDigestor(task="agente")
        result = d._match_tool("guarda una nota sobre el trabajo", TOOLS)
        assert result is None or isinstance(result, dict)

    def test_match_tool_con_texto_vacio(self):
        """_match_tool con texto vacío devuelve herramienta o None sin crashear."""
        d = DataDigestor(task="agente")
        result = d._match_tool("", TOOLS)
        # Puede devolver cualquier herramienta (score 0 en todas) o None
        assert result is None or isinstance(result, dict)

    def test_sample_args_devuelve_dict(self):
        """_sample_args devuelve dict con todas las claves del schema."""
        import random as _r
        d = DataDigestor(task="agente")
        rng = _r.Random(42)
        args = d._sample_args(TOOLS[0], rng)
        assert isinstance(args, dict)
        for key in TOOLS[0]["parameters"]:
            assert key in args

    def test_sample_args_tipos_correctos(self):
        """_sample_args respeta los tipos del schema."""
        import random as _r
        d = DataDigestor(task="agente")
        rng = _r.Random(42)
        tools_typed = [{
            "name": "typed_tool",
            "description": "Test",
            "parameters": {
                "s": {"type": "str"},
                "i": {"type": "int"},
                "f": {"type": "float"},
                "b": {"type": "bool"},
                "l": {"type": "list"},
            }
        }]
        args = d._sample_args(tools_typed[0], rng)
        assert isinstance(args["s"], str)
        assert isinstance(args["i"], int)
        assert isinstance(args["f"], float)
        assert isinstance(args["b"], bool)
        assert isinstance(args["l"], list)

    def test_generate_tool_requests_n_elementos(self):
        """_generate_tool_requests devuelve exactamente n strings."""
        import random as _r
        d = DataDigestor(task="agente")
        rng = _r.Random(42)
        requests = d._generate_tool_requests(TOOLS[0], 7, rng)
        assert len(requests) == 7
        assert all(isinstance(r, str) for r in requests)


# ===========================================================================
# G. Integración con to_jsonl()
# ===========================================================================

class TestIntegracion:

    def test_to_jsonl_exporta_correctamente(self, tmp_path):
        """Tras generate_tool_calls(), to_jsonl() escribe el JSONL."""
        out = tmp_path / "tool_dataset.jsonl"
        d = DataDigestor(task="agente")
        # deduplicate=False para que la dedup no afecte el conteo esperado
        n = d.generate_tool_calls(TOOLS, n_per_tool=3).to_jsonl(str(out), deduplicate=False)
        assert out.exists()
        assert n == len(TOOLS) * 3
        lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == n

    def test_semaforo_despues_de_generar(self):
        """validate() funciona tras generate_tool_calls()."""
        d = DataDigestor(task="agente")
        d.generate_tool_calls(TOOLS, n_per_tool=70)
        report = d.validate(verbose=False)
        assert report["total"] == len(TOOLS) * 70
        assert report["semaforo"] in ("ROJO", "AMARILLO", "VERDE")

    def test_combine_from_api_spec_and_generate_tool_calls(self, tmp_path):
        """from_api_spec + generate_tool_calls acumula ambos bloques."""
        # Crear un spec mínimo
        spec_path = tmp_path / "openapi.yaml"
        spec_path.write_text(
            "openapi: '3.0'\ninfo:\n  title: Test\n  version: '1.0'\n"
            "paths:\n  /ping:\n    get:\n      summary: Ping\n      responses:\n        '200': {description: OK}\n",
            encoding="utf-8",
        )
        d = DataDigestor(task="agente")
        d.from_api_spec(str(spec_path))
        base_count = len(d._examples)
        d.generate_tool_calls(TOOLS_MINIMAL, n_per_tool=3)
        assert len(d._examples) == base_count + 3
