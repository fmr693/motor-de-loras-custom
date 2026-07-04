"""
tests/test_session_15mayo.py
============================
Suite de tests para todo lo implementado el 15 mayo 2026.

Cubre (sin GPU, sin conexión a red):
  A. to_multiturn()         — formato ChatML multi-turno real
  B. _parse_output rescate  — raw_input → JSON correcto
  C. _SYSTEM_AGENT          — placeholders work_dir, tools, max_steps
  D. _fuzzy_path / _DIR_ALIASES — bilingüe ES↔EN
  E. DataDigestor.from_api_spec — OpenAPI 3.x dict + JSONL output
  F. ContinualLearner       — replay buffer, registro, rollback, history
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path



# ===========================================================================
# A — to_multiturn()
# ===========================================================================

class TestToMultiturn(unittest.TestCase):
    """to_multiturn genera la secuencia system/user/assistant alternada correcta."""

    def _example(self, n_steps: int = 2):
        from motor.domestic_dataset_gen import DomesticExample, Step
        steps: list[Step] = [
            {
                "thought":  f"Pensamiento del paso {i+1}",
                "tool":     "search_files",
                "args":     {"query": "factura", "path": "~"},
                "result":   f"Resultado del paso {i+1}",
            }
            for i in range(n_steps)
        ]
        return {
            "user_request":  "Busca las facturas del último mes",
            "steps":         steps,
            "final_thought": "Todo listo.",
            "final_answer":  "Encontré 3 facturas.",
            "language":      "es",
            "category":      "search_files",
        }

    def test_output_has_messages_key(self):
        from motor.domestic_dataset_gen import to_multiturn
        out = to_multiturn(self._example())
        self.assertIn("messages", out)

    def test_first_message_is_system(self):
        from motor.domestic_dataset_gen import to_multiturn
        msgs = to_multiturn(self._example())["messages"]
        self.assertEqual(msgs[0]["role"], "system")

    def test_second_message_is_user_task(self):
        from motor.domestic_dataset_gen import to_multiturn
        msgs = to_multiturn(self._example())["messages"]
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("Tarea:", msgs[1]["content"])

    def test_alternating_assistant_user_per_step(self):
        """Por cada step: assistant (Thought+Action) → user (Observation)."""
        from motor.domestic_dataset_gen import to_multiturn
        n = 3
        msgs = to_multiturn(self._example(n_steps=n))["messages"]
        # system + user_task + (assistant + user) * n + assistant_final
        expected_len = 2 + n * 2 + 1
        self.assertEqual(len(msgs), expected_len)

    def test_observation_contains_step_result(self):
        from motor.domestic_dataset_gen import to_multiturn
        msgs = to_multiturn(self._example(n_steps=1))["messages"]
        # msgs[2]=assistant, msgs[3]=user(Observation)
        obs = msgs[3]["content"]
        self.assertTrue(obs.startswith("Observation:"))
        self.assertIn("Resultado del paso 1", obs)

    def test_last_message_is_final_answer(self):
        from motor.domestic_dataset_gen import to_multiturn
        msgs = to_multiturn(self._example())["messages"]
        last = msgs[-1]
        self.assertEqual(last["role"], "assistant")
        self.assertIn("Final Answer:", last["content"])
        self.assertIn("3 facturas", last["content"])

    def test_action_input_in_intermediate_assistant_is_valid_json(self):
        """El Action Input del assistant intermedio debe ser JSON parseable."""
        from motor.domestic_dataset_gen import to_multiturn
        msgs = to_multiturn(self._example(n_steps=2))["messages"]
        # msgs[2] = primer assistant
        content = msgs[2]["content"]
        m = __import__("re").search(r"Action Input:\s*(\{.+\})", content)
        self.assertIsNotNone(m, "No se encontró Action Input en assistant intermedio")
        try:
            json.loads(m.group(1))
        except json.JSONDecodeError as e:
            self.fail(f"Action Input no es JSON válido: {e}")

    def test_chatml_multiturn_mode(self):
        """to_chatml con renderer='multiturn' devuelve el mismo resultado que to_multiturn."""
        from motor.domestic_dataset_gen import to_chatml, to_multiturn
        ex = self._example()
        direct = to_multiturn(ex)
        via_chatml = to_chatml(ex, renderer="multiturn")
        self.assertEqual(direct, via_chatml)


# ===========================================================================
# B — _parse_output rescate raw_input
# ===========================================================================

class TestParseOutputRescue(unittest.TestCase):
    """_parse_output rescata {"raw_input": "..."} convirtiéndolo al JSON real."""

    def _parse(self, text: str):
        """Invoca _parse_output sin instanciar LoRAAgent (usa método estático-like)."""
        from motor.agent import LoRAAgent

        # Creamos instancia mínima parcheando el constructor para evitar GPU
        agent = object.__new__(LoRAAgent)
        agent.max_steps = 10
        agent._tool_map = {}
        agent._TOOL_ALIASES = {}
        return agent._parse_output(text)

    def test_raw_input_json_rescued(self):
        """Si Action Input es {"raw_input": "{...}"}, parsea el valor interno."""
        text = (
            'Thought: Voy a buscar\n'
            'Action: search_files\n'
            'Action Input: {"raw_input": "{\\"query\\": \\"factura\\", \\"path\\": \\"~\\"}"}'
        )
        _, action, action_input, is_final, _ = self._parse(text)
        self.assertEqual(action, "search_files")
        self.assertIsNotNone(action_input)
        self.assertIn("query", action_input)
        self.assertEqual(action_input["query"], "factura")

    def test_raw_input_shell_converted(self):
        """raw_input con acción shell → {"command": valor}."""
        text = (
            'Thought: Ejecuto ls\n'
            'Action: shell\n'
            'Action Input: {"raw_input": "ls ~"}'
        )
        _, action, action_input, is_final, _ = self._parse(text)
        self.assertEqual(action, "shell")
        self.assertEqual(action_input, {"command": "ls ~"})

    def test_valid_json_not_modified(self):
        """Action Input JSON válido no debe ser modificado."""
        text = (
            'Thought: Ok\n'
            'Action: note_save\n'
            'Action Input: {"title": "Mi nota", "content": "Hola"}'
        )
        _, _, action_input, _, _ = self._parse(text)
        self.assertEqual(action_input["title"], "Mi nota")
        self.assertEqual(action_input["content"], "Hola")

    def test_final_answer_detected(self):
        text = "Thought: Listo\nFinal Answer: He organizado los archivos."
        _, _, _, is_final, final_answer = self._parse(text)
        self.assertTrue(is_final)
        self.assertEqual(final_answer, "He organizado los archivos.")

    def test_final_answer_priority_over_action(self):
        """Si hay Final Answer Y Action, Final Answer tiene prioridad."""
        text = (
            "Thought: Terminé\n"
            "Action: file_organize\n"
            "Action Input: {}\n"
            "Final Answer: Todo hecho."
        )
        _, action, _, is_final, final_answer = self._parse(text)
        self.assertTrue(is_final)
        self.assertIsNone(action)
        self.assertEqual(final_answer, "Todo hecho.")


# ===========================================================================
# C — _SYSTEM_AGENT placeholders
# ===========================================================================

class TestSystemAgentPrompt(unittest.TestCase):
    """El system prompt tiene los 3 placeholders y se formatea sin KeyError."""

    def test_system_agent_has_work_dir_placeholder(self):
        from motor.agent import _SYSTEM_AGENT
        self.assertIn("{work_dir}", _SYSTEM_AGENT)

    def test_system_agent_has_tools_placeholder(self):
        from motor.agent import _SYSTEM_AGENT
        self.assertIn("{tools}", _SYSTEM_AGENT)

    def test_system_agent_has_max_steps_placeholder(self):
        from motor.agent import _SYSTEM_AGENT
        self.assertIn("{max_steps}", _SYSTEM_AGENT)

    def test_system_agent_formats_without_error(self):
        """format() con los 3 valores no lanza KeyError."""
        from motor.agent import _SYSTEM_AGENT
        try:
            result = _SYSTEM_AGENT.format(tools="tool_a, tool_b", max_steps=10, work_dir="~")
        except KeyError as e:
            self.fail(f"_SYSTEM_AGENT.format() lanzó KeyError: {e}")
        self.assertIn("~", result)
        self.assertIn("tool_a", result)

    def test_system_agent_json_example_in_prompt(self):
        """El prompt contiene el ejemplo CORRECTO de JSON."""
        from motor.agent import _SYSTEM_AGENT
        self.assertIn("CORRECTO", _SYSTEM_AGENT)
        self.assertIn("INCORRECTO", _SYSTEM_AGENT)

    def test_domestic_system_uses_work_dir(self):
        """_build_domestic_system no lanza KeyError al formatearse."""
        from motor.domestic_dataset_gen import _build_domestic_system
        try:
            system = _build_domestic_system()
        except KeyError as e:
            self.fail(f"_build_domestic_system() lanzó KeyError: {e}")
        self.assertTrue(len(system) > 100)


# ===========================================================================
# D — _fuzzy_path / _DIR_ALIASES
# ===========================================================================

class TestFuzzyPath(unittest.TestCase):
    """_fuzzy_path resuelve aliases bilingüe ES↔EN si la ruta no existe."""

    def test_existing_path_returned_as_is(self):
        """Si la ruta existe, la devuelve sin modificar."""
        from motor.domestic_tools import _fuzzy_path
        home = Path.home()
        result = _fuzzy_path(str(home))
        self.assertEqual(result.resolve(), home.resolve())

    def test_dir_aliases_not_empty(self):
        from motor.domestic_tools import _DIR_ALIASES
        self.assertGreater(len(_DIR_ALIASES), 0)

    def test_dir_aliases_bidirectional(self):
        """Desktop↔Escritorio deben estar en los aliases."""
        from motor.domestic_tools import _DIR_ALIASES
        all_keys = list(_DIR_ALIASES.keys())
        all_values_flat = [v for vals in _DIR_ALIASES.values() for v in vals]
        self.assertTrue(
            "Desktop" in all_keys or "Desktop" in all_values_flat,
            "Desktop no está en _DIR_ALIASES"
        )
        self.assertTrue(
            "Escritorio" in all_keys or "Escritorio" in all_values_flat,
            "Escritorio no está en _DIR_ALIASES"
        )

    def test_nonexistent_path_fallback_to_parent(self):
        """Ruta que no existe → devuelve el directorio o un Path válido."""
        from motor.domestic_tools import _fuzzy_path
        result = _fuzzy_path("~/CarpetaQueNoExisteJamas_ZZZ999")
        # Debe devolver un Path, no lanzar excepción
        self.assertIsInstance(result, Path)

    def test_fuzzy_path_with_real_home_subfolder(self):
        """Si ~/Documents existe, _fuzzy_path('~/Documents') lo resuelve."""
        from motor.domestic_tools import _fuzzy_path
        docs = Path.home() / "Documents"
        if docs.exists():
            result = _fuzzy_path("~/Documents")
            self.assertTrue(result.exists())

    def test_search_files_uses_fuzzy_path(self):
        """search_files con ruta inexistente no lanza excepción (usa tmpdir controlado)."""
        import tempfile
        from motor.domestic_tools import search_files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Directorio vacío sin coincidencias — debe responder rápido
            result = search_files("*.txt", tmpdir)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)


# ===========================================================================
# E — DataDigestor.from_api_spec()
# ===========================================================================

class TestFromApiSpec(unittest.TestCase):
    """from_api_spec genera ejemplos ChatML a partir de spec OpenAPI."""

    def _mini_spec(self) -> dict:
        return {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0"},
            "servers": [{"url": "https://api.test.com"}],
            "paths": {
                "/items": {
                    "get": {
                        "summary": "Lista todos los items disponibles",
                        "parameters": [
                            {"name": "limit",  "in": "query", "schema": {"type": "integer", "example": 10}},
                            {"name": "filter", "in": "query", "schema": {"type": "string",  "example": "activo"}},
                        ],
                        "responses": {"200": {"description": "Lista de items devuelta."}},
                    },
                    "post": {
                        "summary": "Crea un nuevo item en el sistema",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name":  {"type": "string",  "example": "Widget"},
                                            "price": {"type": "number",  "example": 9.99},
                                        },
                                    }
                                }
                            }
                        },
                        "responses": {"201": {"description": "Item creado correctamente."}},
                    },
                },
                "/items/{id}": {
                    "delete": {
                        "summary": "Elimina un item por su ID",
                        "parameters": [
                            {"name": "id", "in": "path", "schema": {"type": "integer", "example": 1}},
                        ],
                        "responses": {"200": {"description": "Item eliminado."}},
                    },
                },
            },
        }

    def setUp(self):
        from motor.digestor import DataDigestor
        self.d = DataDigestor("Llama al endpoint correcto de la API")

    def test_returns_self_for_chaining(self):
        result = self.d.from_api_spec(self._mini_spec())
        self.assertIs(result, self.d)

    def test_generates_one_example_per_endpoint(self):
        self.d.from_api_spec(self._mini_spec())
        reg = self.d.get_registry() if hasattr(self.d, "get_registry") else None
        # 3 endpoints → 3 ejemplos
        self.assertEqual(len(self.d._examples), 3)

    def test_each_example_has_messages(self):
        self.d.from_api_spec(self._mini_spec())
        for ex in self.d._examples:
            self.assertIn("messages", ex)

    def test_messages_have_three_roles(self):
        self.d.from_api_spec(self._mini_spec())
        for ex in self.d._examples:
            roles = [m["role"] for m in ex["messages"]]
            self.assertIn("system", roles)
            self.assertIn("user", roles)
            self.assertIn("assistant", roles)

    def test_assistant_contains_thought_and_action(self):
        self.d.from_api_spec(self._mini_spec())
        for ex in self.d._examples:
            assistant = next(m for m in ex["messages"] if m["role"] == "assistant")
            self.assertIn("Thought:", assistant["content"])
            self.assertIn("Action:", assistant["content"])

    def test_assistant_contains_final_answer(self):
        self.d.from_api_spec(self._mini_spec())
        for ex in self.d._examples:
            assistant = next(m for m in ex["messages"] if m["role"] == "assistant")
            self.assertIn("Final Answer:", assistant["content"])

    def test_action_input_is_valid_json(self):
        """El Action Input en el assistant debe ser JSON válido (soporta objetos anidados)."""
        import re
        from motor.agent import _extract_json_object
        self.d.from_api_spec(self._mini_spec())
        for ex in self.d._examples:
            assistant = next(m for m in ex["messages"] if m["role"] == "assistant")
            content = assistant["content"]
            ai_match = re.search(r"Action Input:\s*\{", content, re.IGNORECASE)
            if ai_match:
                raw = _extract_json_object(content, ai_match.end() - 1)
                if raw:
                    try:
                        json.loads(raw)
                    except json.JSONDecodeError as e:
                        self.fail(f"Action Input no es JSON válido: {e}\n{raw}")

    def test_function_call_format(self):
        """format='function_call' genera mensajes con tool_call_content."""
        d2 = self.d.__class__("Test")
        d2.from_api_spec(self._mini_spec(), format="function_call")
        self.assertEqual(len(d2._examples), 3)
        for ex in d2._examples:
            assistant = next(m for m in ex["messages"] if m["role"] == "assistant")
            self.assertIn("<tool_call>", assistant["content"])

    def test_n_limits_endpoints(self):
        """n=1 genera exactamente 1 ejemplo."""
        d2 = self.d.__class__("Test")
        d2.from_api_spec(self._mini_spec(), n=1)
        self.assertEqual(len(d2._examples), 1)

    def test_to_jsonl_works_after_from_api_spec(self):
        """El output es compatible con to_jsonl()."""
        self.d.from_api_spec(self._mini_spec())
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out = f.name
        try:
            n = self.d.to_jsonl(out)
            self.assertEqual(n, 3)
            lines = Path(out).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            for line in lines:
                parsed = json.loads(line)
                self.assertIn("messages", parsed)
        finally:
            os.unlink(out)

    def test_unknown_format_raises(self):
        d2 = self.d.__class__("Test")
        with self.assertRaises(ValueError):
            d2.from_api_spec(self._mini_spec(), format="unknown_format")

    def test_file_not_found_raises(self):
        d2 = self.d.__class__("Test")
        with self.assertRaises(FileNotFoundError):
            d2.from_api_spec("/ruta/que/no/existe/spec.json")


# ===========================================================================
# F — ContinualLearner
# ===========================================================================

class TestContinualLearner(unittest.TestCase):
    """ContinualLearner: registro, replay buffer, detección de regresión, rollback."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.reg_path = str(Path(self.tmpdir) / "registry.json")
        from motor.continual import ContinualLearner
        self.cl = ContinualLearner(
            model_id           = "Qwen/Qwen2.5-3B-Instruct",
            registry_path      = self.reg_path,
            replay_buffer_size = 10,
            rollback_threshold = 0.15,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_dataset(self, n: int, name: str) -> Path:
        path = Path(self.tmpdir) / f"{name}.jsonl"
        lines = [
            json.dumps({"messages": [
                {"role": "user",      "content": f"pregunta {i}"},
                {"role": "assistant", "content": f"respuesta {i}"},
            ]})
            for i in range(n)
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _make_adapter_dir(self, name: str, eval_loss: float) -> Path:
        d = Path(self.tmpdir) / name
        d.mkdir()
        (d / "meta.json").write_text(
            json.dumps({"eval_loss": eval_loss, "train_loss": 0.2}),
            encoding="utf-8",
        )
        return d

    # --- Registro ---

    def test_register_existing_creates_entry(self):
        ds = self._make_dataset(20, "ds_a")
        ad = self._make_adapter_dir("adapter_a", eval_loss=0.42)
        self.cl.register_existing(str(ad), str(ds), name="tarea_a")
        reg = self.cl.get_registry()
        self.assertEqual(len(reg["adapters"]), 1)
        self.assertEqual(reg["adapters"][0]["name"], "tarea_a")
        self.assertAlmostEqual(reg["adapters"][0]["eval_loss"], 0.42)

    def test_registry_persisted_to_disk(self):
        ds = self._make_dataset(10, "ds_b")
        ad = self._make_adapter_dir("adapter_b", eval_loss=0.3)
        self.cl.register_existing(str(ad), str(ds), name="tarea_b")
        # Recargar desde disco
        from motor.continual import ContinualLearner
        cl2 = ContinualLearner("model", registry_path=self.reg_path)
        reg = cl2.get_registry()
        self.assertEqual(len(reg["adapters"]), 1)
        self.assertEqual(reg["adapters"][0]["name"], "tarea_b")

    def test_history_does_not_crash_when_empty(self):
        """history() no lanza excepción con registro vacío."""
        try:
            self.cl.history()
        except Exception as e:
            self.fail(f"history() lanzó excepción: {e}")

    def test_history_does_not_crash_with_entries(self):
        ds = self._make_dataset(10, "ds_h")
        ad = self._make_adapter_dir("adapter_h", eval_loss=0.5)
        self.cl.register_existing(str(ad), str(ds), name="historia_test")
        try:
            self.cl.history()
        except Exception as e:
            self.fail(f"history() lanzó excepción: {e}")

    # --- Replay buffer ---

    def test_replay_buffer_empty_with_no_history(self):
        samples = self.cl._sample_replay_buffer(seed=42)
        self.assertEqual(samples, [])

    def test_replay_buffer_samples_from_past_datasets(self):
        ds_a = self._make_dataset(50, "ds_replay_a")
        ds_b = self._make_dataset(50, "ds_replay_b")
        ad_a = self._make_adapter_dir("ad_rp_a", 0.4)
        ad_b = self._make_adapter_dir("ad_rp_b", 0.35)
        self.cl.register_existing(str(ad_a), str(ds_a), name="rp_a")
        self.cl.register_existing(str(ad_b), str(ds_b), name="rp_b")
        samples = self.cl._sample_replay_buffer(seed=42)
        self.assertGreater(len(samples), 0)
        self.assertLessEqual(len(samples), self.cl.replay_buffer_size)

    def test_build_merged_no_history_returns_original(self):
        ds = self._make_dataset(20, "ds_orig")
        merged, count = self.cl._build_merged_dataset(ds, seed=42)
        self.assertEqual(merged, ds)
        self.assertEqual(count, 0)

    def test_build_merged_with_history_creates_temp_file(self):
        ds_past = self._make_dataset(50, "ds_past_mg")
        ad_past = self._make_adapter_dir("ad_past_mg", 0.4)
        self.cl.register_existing(str(ad_past), str(ds_past), name="past_mg")
        ds_new  = self._make_dataset(20, "ds_new_mg")
        merged, count = self.cl._build_merged_dataset(ds_new, seed=42)
        try:
            self.assertNotEqual(merged, ds_new)
            self.assertGreater(count, 0)
            self.assertTrue(merged.exists())
            lines = merged.read_text(encoding="utf-8").splitlines()
            # El merged tiene más líneas que el nuevo solo (20 + replay)
            self.assertGreater(len(lines), 20)
        finally:
            if merged.exists():
                merged.unlink()

    def test_replay_buffer_size_zero_disables_replay(self):
        from motor.continual import ContinualLearner
        cl_no_replay = ContinualLearner(
            "model", registry_path=self.reg_path, replay_buffer_size=0
        )
        ds = self._make_dataset(20, "ds_zero")
        ad = self._make_adapter_dir("ad_zero", 0.3)
        cl_no_replay.register_existing(str(ad), str(ds), name="zero_test")
        samples = cl_no_replay._sample_replay_buffer(seed=42)
        self.assertEqual(samples, [])

    # --- Detección de regresión ---

    def test_no_regression_without_baseline(self):
        """Sin baseline, _check_regression devuelve (None, False)."""
        pct, triggered = self.cl._check_regression(
            new_eval_loss=0.5,
            adapter_name="nuevo",
            output_dir=Path(self.tmpdir),
            backup_dir=None,
        )
        self.assertIsNone(pct)
        self.assertFalse(triggered)

    def test_no_rollback_below_threshold(self):
        """Regresión del 10% con umbral 15% → NO dispara rollback."""
        ds = self._make_dataset(20, "ds_base")
        ad = self._make_adapter_dir("ad_base", 0.40)
        self.cl.register_existing(str(ad), str(ds), name="tarea_base")
        # 0.44 vs 0.40 = +10% < 15% → sin rollback
        pct, triggered = self.cl._check_regression(
            new_eval_loss=0.44,
            adapter_name="tarea_base",
            output_dir=Path(self.tmpdir),
            backup_dir=None,
        )
        self.assertAlmostEqual(pct, 0.10, places=2)
        self.assertFalse(triggered)

    def test_regression_above_threshold_no_backup(self):
        """Regresión >15% sin backup → triggered=False (no puede revertir)."""
        ds = self._make_dataset(20, "ds_reg")
        ad = self._make_adapter_dir("ad_reg", 0.40)
        self.cl.register_existing(str(ad), str(ds), name="tarea_reg")
        # 0.50 vs 0.40 = +25% > 15% → quiere rollback pero no hay backup
        pct, triggered = self.cl._check_regression(
            new_eval_loss=0.50,
            adapter_name="tarea_reg",
            output_dir=Path(self.tmpdir) / "ad_destino",
            backup_dir=None,
        )
        self.assertGreater(pct, 0.15)
        self.assertFalse(triggered)

    def test_regression_above_threshold_with_backup(self):
        """Regresión >15% CON backup → triggered=True y el adapter se restaura."""
        import shutil
        ds = self._make_dataset(20, "ds_backup")
        ad = self._make_adapter_dir("ad_backup_orig", 0.40)
        self.cl.register_existing(str(ad), str(ds), name="tarea_backup")

        # Simular nuevo adapter (con peor loss) y su backup
        new_ad = Path(self.tmpdir) / "nuevo_adapter"
        new_ad.mkdir()
        (new_ad / "bad_weights.bin").write_bytes(b"bad")
        backup = Path(self.tmpdir) / "nuevo_adapter_backup"
        shutil.copytree(ad, backup)  # backup = copia del adapter bueno

        pct, triggered = self.cl._check_regression(
            new_eval_loss=0.55,  # +37.5% > 15%
            adapter_name="tarea_backup",
            output_dir=new_ad,
            backup_dir=backup,
        )
        self.assertTrue(triggered)
        # El adapter restaurado debe tener meta.json (del backup, no bad_weights.bin)
        self.assertTrue((new_ad / "meta.json").exists())
        self.assertFalse((new_ad / "bad_weights.bin").exists())

    # --- Rollback manual ---

    def test_rollback_no_backup_returns_false(self):
        result = self.cl.rollback(str(Path(self.tmpdir) / "adapter_sin_backup"))
        self.assertFalse(result)

    def test_rollback_with_backup_returns_true(self):
        import shutil
        ad      = self._make_adapter_dir("ad_rbk", 0.3)
        backup  = Path(str(ad) + "_backup")
        shutil.copytree(ad, backup)
        # Corromper el original
        (ad / "meta.json").write_text("{}")
        result = self.cl.rollback(str(ad))
        self.assertTrue(result)
        # El original debe haberse restaurado desde backup (eval_loss=0.3)
        restored = json.loads((ad / "meta.json").read_text())
        self.assertAlmostEqual(restored["eval_loss"], 0.3)

    def test_get_baseline_uses_same_name_first(self):
        """_get_baseline_loss prioriza la entrada con el mismo nombre."""
        ds = self._make_dataset(10, "ds_bl")
        ad = self._make_adapter_dir("ad_bl", 0.42)
        self.cl.register_existing(str(ad), str(ds), name="tarea_especifica")
        # Añadir otro adapter con diferente nombre
        ad2 = self._make_adapter_dir("ad_bl2", 0.99)
        ds2 = self._make_dataset(10, "ds_bl2")
        self.cl.register_existing(str(ad2), str(ds2), name="otra_tarea")
        # El baseline de "tarea_especifica" debe ser 0.42, no 0.99
        baseline = self.cl._get_baseline_loss("tarea_especifica")
        self.assertAlmostEqual(baseline, 0.42)


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
