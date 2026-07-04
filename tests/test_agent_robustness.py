"""
tests/test_agent_robustness.py
==============================
Suite de regresión para el agente ReAct (motor.agent).

Cubre (sin GPU, sin red):
  A. _sanitize_output        — detección de templates sin rellenar, alucinaciones, vacíos
  B. _sanitize_observation   — versión laxa (solo doble llave y vacío)
  C. _extract_json_object    — balance de llaves, strings, escapes
  D. _parse_output           — Thought / Action / Final Answer / raw_input rescue
  E. Tool aliases            — 35+ alias → canónico
  F. System prompt           — placeholders, ejemplos JSON correcto/incorrecto
  G. LoRAAgent.run()         — integración con mock de inferencia
  H. Herramientas            — read_file, list_dir, shell, http_get
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


from motor.agent import (
    _sanitize_output,
    _sanitize_observation,
    _extract_json_object,
    _SHELL_BLOCKED,
    _SYSTEM_AGENT,
    AgentResult,
    AgentStep,
    DEFAULT_TOOLS,
    LoRAAgent,
    Tool,
    _tool_read_file,
    _tool_list_dir,
    _tool_shell,
    _tool_http_get,
)


# ===========================================================================
# A — _sanitize_output
# ===========================================================================

class TestSanitizeOutput(unittest.TestCase):
    """Detección de templates sin rellenar, alucinaciones y respuestas vacías."""

    # ── Casos válidos (debe devolver None) ──────────────────────────

    def test_texto_normal_es_valido(self):
        self.assertIsNone(_sanitize_output("He movido 3 archivos a Documentos/Facturas."))

    def test_texto_largo_es_valido(self):
        self.assertIsNone(_sanitize_output("A" * 200))

    def test_texto_con_llaves_json_es_valido(self):
        """JSON legítimo no debe disparar falsos positivos."""
        self.assertIsNone(_sanitize_output('La acción fue {"command": "ls ~"}'))

    def test_texto_con_parentesis_es_valido(self):
        self.assertIsNone(_sanitize_output("El archivo (factura.pdf) fue movido."))

    def test_texto_con_corchetes_normales_es_valido(self):
        """Corchetes normales [como este] no deben disparar."""
        self.assertIsNone(_sanitize_output("Encontré [factura_ene.pdf, factura_feb.pdf] en la carpeta."))

    # ── Templates sin rellenar ──────────────────────────────────────

    def test_detecta_doble_llave(self):
        motivo = _sanitize_output("El archivo {{nombre}} fue procesado.")
        self.assertIsNotNone(motivo)
        self.assertIn("template sin rellenar", motivo)

    def test_detecta_placeholder_mayusculas_llaves(self):
        motivo = _sanitize_output("Resultado: {NOMBRE} no encontrado.")
        self.assertIsNotNone(motivo)
        self.assertIn("template sin rellenar", motivo)

    def test_detecta_placeholder_mayusculas_corchetes(self):
        motivo = _sanitize_output("El valor es [RESULTADO].")
        self.assertIsNotNone(motivo)
        self.assertIn("template sin rellenar", motivo)

    def test_detecta_placeholder_literal(self):
        motivo = _sanitize_output("Aquí va el [placeholder] del resultado.")
        self.assertIsNotNone(motivo)
        self.assertIn("template sin rellenar", motivo)

    def test_detecta_placeholder_html(self):
        motivo = _sanitize_output("El comando <NOMBRE> no se reconoce.")
        self.assertIsNotNone(motivo)
        self.assertIn("template sin rellenar", motivo)

    # ── Alucinaciones ───────────────────────────────────────────────

    def test_detecta_no_tengo_informacion(self):
        motivo = _sanitize_output("No tengo información sobre ese archivo.")
        self.assertIsNotNone(motivo)
        self.assertIn("posible alucinación", motivo)

    def test_detecta_como_modelo_de_lenguaje(self):
        motivo = _sanitize_output("Como modelo de lenguaje, no puedo acceder a archivos.")
        self.assertIsNotNone(motivo)
        self.assertIn("posible alucinación", motivo)

    def test_detecta_no_se_puede_determinar(self):
        motivo = _sanitize_output("No se puede determinar la ubicación del archivo.")
        self.assertIsNotNone(motivo)
        self.assertIn("posible alucinación", motivo)

    def test_detecta_no_se_puede_saber(self):
        motivo = _sanitize_output("No se puede saber con certeza.")
        self.assertIsNotNone(motivo)
        self.assertIn("posible alucinación", motivo)

    # ── Vacíos / cortos ─────────────────────────────────────────────

    def test_respuesta_vacia(self):
        motivo = _sanitize_output("")
        self.assertIsNotNone(motivo)
        self.assertIn("vacía", motivo)

    def test_respuesta_solo_espacios(self):
        motivo = _sanitize_output("   ")
        self.assertIsNotNone(motivo)
        self.assertIn("vacía", motivo)

    def test_respuesta_demasiado_corta(self):
        motivo = _sanitize_output("OK")
        self.assertIsNotNone(motivo)
        self.assertIn("corta", motivo)

    def test_none_es_vacio(self):
        motivo = _sanitize_output(None)
        self.assertIsNotNone(motivo)


# ===========================================================================
# B — _sanitize_observation
# ===========================================================================

class TestSanitizeObservation(unittest.TestCase):
    """Versión laxa: solo vacío total o doble llave {{...}}."""

    def test_observacion_normal_es_valida(self):
        self.assertIsNone(_sanitize_observation("Archivo leído correctamente."))

    def test_observacion_con_json_es_valida(self):
        """JSON de tool no debe disparar falso positivo."""
        self.assertIsNone(_sanitize_observation('{"files": ["a.pdf"], "moved": 1}'))

    def test_observacion_con_llaves_simples_es_valida(self):
        """Llaves simples en output de tool son normales."""
        self.assertIsNone(_sanitize_observation("C:/Users/Felipe/Desktop/"))

    def test_observacion_con_corchetes_es_valida(self):
        self.assertIsNone(_sanitize_observation("  factura.pdf  (1234 B)"))

    def test_observacion_vacia_es_invalida(self):
        motivo = _sanitize_observation("")
        self.assertIsNotNone(motivo)
        self.assertIn("vacía", motivo)

    def test_observacion_con_doble_llave_es_invalida(self):
        motivo = _sanitize_observation("Resultado: {{resultado}} no encontrado")
        self.assertIsNotNone(motivo)
        self.assertIn("template sin rellenar", motivo)


# ===========================================================================
# C — _extract_json_object
# ===========================================================================

class TestExtractJsonObject(unittest.TestCase):
    """Extracción de JSON con balance de llaves."""

    def test_json_simple(self):
        text = 'Action Input: {"command": "ls -la"}'
        result = _extract_json_object(text, text.index("{"))
        self.assertEqual(result, '{"command": "ls -la"}')

    def test_json_con_llaves_dentro_de_string(self):
        """awk '{print $2}' contiene llaves — no deben romper el balance."""
        text = 'Action Input: {"command": "grep error | awk \'{print $2}\'"}'
        start = text.index("{")
        result = _extract_json_object(text, start)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertIn("awk", parsed["command"])

    def test_json_anidado(self):
        text = '{"outer": {"inner": {"deep": "value"}}}'
        result = _extract_json_object(text, 0)
        self.assertEqual(result, text)

    def test_json_con_escape_backslash(self):
        text = '{"path": "C:\\\\Users\\\\Felipe"}'
        result = _extract_json_object(text, 0)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertIn("Felipe", parsed["path"])

    def test_json_con_comillas_escapadas_en_string(self):
        text = '{"msg": "ella dijo \\"hola\\" ayer"}'
        result = _extract_json_object(text, 0)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertIn('"hola"', parsed["msg"])

    def test_json_malformado_devuelve_none(self):
        text = '{"command": "ls"'
        start = text.index("{")
        result = _extract_json_object(text, start)
        self.assertIsNone(result)

    def test_varios_json_en_texto_extrae_el_primero(self):
        text = 'Primero: {"a": 1}, segundo: {"b": 2}'
        start = text.index("{")
        result = _extract_json_object(text, start)
        self.assertEqual(result, '{"a": 1}')


# ===========================================================================
# D — _parse_output
# ===========================================================================

class TestParseOutput(unittest.TestCase):
    """Parsing de la respuesta del LLM."""

    @classmethod
    def setUpClass(cls):
        cls.agent = LoRAAgent(infer_fn=lambda *a, **k: "")

    # ── Final Answer ────────────────────────────────────────────────

    def test_final_answer_simple(self):
        text = "Thought: Ya tengo la información.\nFinal Answer: Encontré 3 archivos."
        thought, action, action_input, is_final, final_answer = self.agent._parse_output(text)
        self.assertTrue(is_final)
        self.assertEqual(final_answer, "Encontré 3 archivos.")
        self.assertIn("Ya tengo", thought)

    def test_final_answer_multilinea(self):
        text = (
            "Thought: Todo listo.\n"
            "Final Answer: Resultados:\n"
            "- archivo1.pdf\n"
            "- archivo2.pdf\n"
        )
        _, _, _, is_final, final_answer = self.agent._parse_output(text)
        self.assertTrue(is_final)
        self.assertIn("archivo1.pdf", final_answer)

    # ── Action ──────────────────────────────────────────────────────

    def test_action_con_json_correcto(self):
        text = (
            'Thought: Necesito listar el directorio.\n'
            'Action: list_dir\n'
            'Action Input: {"path": "~"}'
        )
        thought, action, action_input, is_final, _ = self.agent._parse_output(text)
        self.assertFalse(is_final)
        self.assertEqual(action, "list_dir")
        self.assertEqual(action_input, {"path": "~"})

    def test_action_con_raw_input_rescatado(self):
        """raw_input rescue: el modelo generó {"raw_input": "ls ~"} → se rescata."""
        text = (
            'Thought: Necesito ver archivos.\n'
            'Action: shell\n'
            'Action Input: {"raw_input": "ls ~/Documentos"}'
        )
        _, action, action_input, _, _ = self.agent._parse_output(text)
        self.assertEqual(action, "shell")
        self.assertEqual(action_input, {"command": "ls ~/Documentos"})

    def test_action_sin_input(self):
        text = "Thought: Puedo responder.\nAction: list_dir"
        _, action, action_input, is_final, _ = self.agent._parse_output(text)
        self.assertEqual(action, "list_dir")
        self.assertIsNone(action_input)
        self.assertFalse(is_final)

    # ── Sin formato ─────────────────────────────────────────────────

    def test_sin_action_ni_final(self):
        text = "Esto es solo un pensamiento sin acción."
        thought, action, _, is_final, _ = self.agent._parse_output(text)
        self.assertFalse(is_final)
        self.assertIsNone(action)

    def test_thought_vacio(self):
        text = "Action: list_dir\nAction Input: {\"path\": \".\"}"
        thought, action, _, _, _ = self.agent._parse_output(text)
        self.assertEqual(action, "list_dir")
        self.assertEqual(thought, "")

    # ── Final Answer tiene prioridad sobre Action ───────────────────

    def test_final_answer_prioridad_sobre_action(self):
        text = (
            "Thought: Ya terminé.\n"
            "Action: list_dir\n"
            "Final Answer: El directorio tiene 5 archivos."
        )
        _, action, _, is_final, final_answer = self.agent._parse_output(text)
        self.assertTrue(is_final)
        self.assertIsNone(action)
        self.assertIn("5 archivos", final_answer)


# ===========================================================================
# E — Tool Aliases
# ===========================================================================

class TestToolAliases(unittest.TestCase):
    """35+ alias → canónico."""

    def test_save_note_se_resuelve(self):
        agent = LoRAAgent(infer_fn=lambda *a, **k: "")
        self.assertEqual(agent._TOOL_ALIASES.get("save_note"), "note_save")
        self.assertEqual(agent._TOOL_ALIASES.get("create_note"), "note_save")

    def test_bash_es_shell(self):
        agent = LoRAAgent(infer_fn=lambda *a, **k: "")
        self.assertEqual(agent._TOOL_ALIASES.get("bash"), "shell")
        self.assertEqual(agent._TOOL_ALIASES.get("cmd"), "shell")
        self.assertEqual(agent._TOOL_ALIASES.get("execute"), "shell")

    def test_get_gpu_se_resuelven_a_shell(self):
        agent = LoRAAgent(infer_fn=lambda *a, **k: "")
        self.assertEqual(agent._TOOL_ALIASES.get("get_gpu_memory"), "shell")
        self.assertEqual(agent._TOOL_ALIASES.get("get_vram"), "shell")
        self.assertEqual(agent._TOOL_ALIASES.get("gpu_info"), "shell")

    def test_alias_no_existente_devuelve_none(self):
        agent = LoRAAgent(infer_fn=lambda *a, **k: "")
        self.assertIsNone(agent._TOOL_ALIASES.get("herramienta_imposible"))

    def test_alias_no_sobrescribe_herramientas_reales(self):
        """Un alias no debe colisionar con una herramienta real."""
        agent = LoRAAgent(infer_fn=lambda *a, **k: "")
        # Si el alias ya es una tool real, no debería estar en aliases
        for alias in agent._TOOL_ALIASES:
            self.assertNotIn(alias, agent.tools, f"Alias '{alias}' colisiona con tool real")


# ===========================================================================
# F — System Prompt
# ===========================================================================

class TestSystemPrompt(unittest.TestCase):
    """Validación del system prompt del agente."""

    def setUp(self):
        self.agent = LoRAAgent(infer_fn=lambda *a, **k: "", work_dir="/home/user/proyecto")

    def test_prompt_contiene_work_dir(self):
        prompt = self.agent._build_system()
        self.assertIn("/home/user/proyecto", prompt)

    def test_prompt_contiene_tools(self):
        prompt = self.agent._build_system()
        for tool_name in ["read_file", "list_dir", "shell", "http_get"]:
            self.assertIn(tool_name, prompt, f"Falta tool {tool_name} en system prompt")

    def test_prompt_contiene_max_steps(self):
        prompt = self.agent._build_system()
        self.assertIn("12", prompt)  # default max_steps

    def test_prompt_contiene_ejemplo_json(self):
        prompt = self.agent._build_system()
        self.assertIn('{"command"', prompt)
        self.assertIn('{"raw_input"', prompt)  # ejemplo INCORRECTO

    def test_prompt_contiene_regla_descubrimiento(self):
        prompt = self.agent._build_system()
        self.assertIn("list_dir", prompt.lower())


# ===========================================================================
# G — LoRAAgent.run() con mock de inferencia
# ===========================================================================

class TestAgentRunWithMock(unittest.TestCase):
    """Integración del agente con inferencia mockeada."""

    def setUp(self):
        # Mock que devuelve un Final Answer correcto
        self.mock_infer = MagicMock(return_value=(
            "Thought: Puedo responder directamente.\n"
            "Final Answer: El workspace contiene 15 archivos Python."
        ))

    def test_respuesta_directa_sin_tools(self):
        agent = LoRAAgent(infer_fn=self.mock_infer, max_steps=5)
        result = agent.run("¿Cuántos archivos Python hay?")
        self.assertTrue(result.success)
        self.assertIn("15 archivos", result.answer)
        self.assertEqual(len(result.steps), 1)
        self.assertTrue(result.steps[0].is_final)

    def test_final_answer_sanitizada_rechazada(self):
        """Si el modelo devuelve template sin rellenar, el agente da error."""
        mock = MagicMock(return_value=(
            "Thought: Ya tengo la información.\n"
            "Final Answer: El resultado es {{valor}}."
        ))
        agent = LoRAAgent(infer_fn=mock, max_steps=5)
        result = agent.run("¿Cuál es el resultado?")
        self.assertFalse(result.success)
        self.assertIn("sanitización", result.answer.lower())
        self.assertIn("{{valor}}", result.answer)

    def test_final_answer_alucinacion_rechazada(self):
        mock = MagicMock(return_value=(
            "Thought: No puedo hacerlo.\n"
            "Final Answer: No tengo información sobre ese archivo."
        ))
        agent = LoRAAgent(infer_fn=mock, max_steps=5)
        result = agent.run("Busca el archivo.")
        self.assertFalse(result.success)
        self.assertIn("sanitización", result.answer.lower())

    def test_final_answer_demasiado_corta_rechazada(self):
        mock = MagicMock(return_value=(
            "Thought: Listo.\nFinal Answer: OK"
        ))
        agent = LoRAAgent(infer_fn=mock, max_steps=5)
        result = agent.run("Haz algo.")
        self.assertFalse(result.success)
        self.assertIn("corta", result.answer.lower())

    def test_sin_formato_recibe_hint(self):
        """Si el modelo no sigue el formato, recibe un hint de corrección."""
        mock = MagicMock(return_value="No sé qué hacer.")
        agent = LoRAAgent(infer_fn=mock, max_steps=3)
        result = agent.run("Tarea cualquiera.")
        # Agota pasos porque nunca da formato correcto
        self.assertFalse(result.success)
        self.assertIn("pasos", result.answer.lower())

    def test_herramienta_desconocida_da_error(self):
        """Si el modelo pide una herramienta que no existe."""
        mock = MagicMock(return_value=(
            "Thought: Necesito una herramienta especial.\n"
            "Action: herramienta_imaginaria\n"
            "Action Input: {\"param\": \"valor\"}"
        ))
        agent = LoRAAgent(infer_fn=mock, max_steps=3)
        result = agent.run("Haz algo imposible.")
        self.assertFalse(result.success)
        self.assertIn("pasos", result.answer.lower())

    def test_observacion_sanitizada_rechazada(self):
        """Si una tool devuelve template sin rellenar, el agente da error."""
        mock = MagicMock(return_value=(
            "Thought: Necesito leer un archivo.\n"
            "Action: read_file\n"
            'Action Input: {"path": "datos.json"}'
        ))
        # Crear una tool que devuelva template sin rellenar
        tool_mala = Tool(
            name="read_file",
            description="Lee archivos",
            params_doc='{"path": "..."}',
            fn=lambda path: "Contenido: {{placeholder}}",
        )
        agent = LoRAAgent(infer_fn=mock, tools=[tool_mala], max_steps=5)
        result = agent.run("Lee el archivo datos.json")
        self.assertFalse(result.success)
        self.assertIn("sanitización", result.answer.lower())

    def test_error_inferencia_capturado(self):
        """Si la inferencia lanza excepción, se captura."""
        mock = MagicMock(side_effect=RuntimeError("GPU out of memory"))
        agent = LoRAAgent(infer_fn=mock, max_steps=5)
        result = agent.run("Cualquier tarea.")
        self.assertFalse(result.success)
        self.assertIn("Error en inferencia", result.answer)

    def test_flujo_multi_paso_exitoso(self):
        """Dos pasos: list_dir → Final Answer."""
        call_count = [0]

        def multi_step_infer(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    "Thought: Primero listo el directorio.\n"
                    'Action: list_dir\n'
                    'Action Input: {"path": "."}'
                )
            else:
                return (
                    "Thought: Ya tengo los datos.\n"
                    "Final Answer: El directorio contiene 3 carpetas y 5 archivos."
                )

        agent = LoRAAgent(infer_fn=multi_step_infer, max_steps=5)
        result = agent.run("¿Qué hay en el directorio actual?")
        self.assertTrue(result.success)
        self.assertIn("3 carpetas", result.answer)
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].action, "list_dir")
        self.assertTrue(result.steps[1].is_final)

    def test_rescate_alias_en_accion(self):
        """Si el modelo dice 'bash' en vez de 'shell', el alias lo resuelve."""
        # Mock: llama a 'bash' en vez de 'shell'
        mock = MagicMock(return_value=(
            "Thought: Necesito ejecutar un comando.\n"
            "Action: bash\n"
            'Action Input: {"command": "ls"}'
        ))
        agent = LoRAAgent(infer_fn=mock, max_steps=3)
        result = agent.run("Lista archivos.")
        # Llega a ejecutar la tool (shell) y devuelve observación
        # Luego el modelo debería dar otro paso, pero como mock es fijo,
        # seguirá pidiendo bash → eventualmente se agotan pasos
        self.assertFalse(result.success)  # se agotan pasos porque mock no da Final Answer
        # Pero al menos el primer paso usó la tool correcta (shell)
        self.assertTrue(any(s.action == "shell" for s in result.steps))


# ===========================================================================
# H — Herramientas (read_file, list_dir, shell, http_get)
# ===========================================================================

class TestTools(unittest.TestCase):
    """Validación directa de las herramientas."""

    # ── read_file ───────────────────────────────────────────────────

    def test_read_file_existente(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("linea 1\nlinea 2\nlinea 3")
            tmp = f.name
        try:
            result = _tool_read_file(tmp)
            self.assertIn("linea 1", result)
        finally:
            os.unlink(tmp)

    def test_read_file_no_existente(self):
        result = _tool_read_file("/ruta/que/no/existe.txt")
        self.assertIn("no encontrado", result)

    def test_read_file_es_directorio(self):
        result = _tool_read_file(str(Path.home()))
        self.assertIn("No es un archivo", result)

    # ── list_dir ────────────────────────────────────────────────────

    def test_list_dir_existente(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "archivo.txt").write_text("contenido")
            (Path(tmp) / "subcarpeta").mkdir()
            result = _tool_list_dir(tmp)
            self.assertIn("archivo.txt", result)
            self.assertIn("subcarpeta/", result)

    def test_list_dir_no_existente(self):
        result = _tool_list_dir("/ruta/que/no/existe")
        self.assertIn("no encontrada", result)

    def test_list_dir_es_archivo(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp = f.name
        try:
            result = _tool_list_dir(tmp)
            self.assertIn("No es un directorio", result)
        finally:
            os.unlink(tmp)

    # ── shell ───────────────────────────────────────────────────────

    def test_shell_comando_permitido(self):
        # _tool_shell usa cwd=~/Proyecto_V3; si no existe falla con WinError 267.
        # Probamos el mecanismo de bloqueo en vez de la ejecución real.
        # El bloqueo de comandos peligrosos es la parte crítica de seguridad.
        result = _tool_shell("echo hola")
        # Si el cwd existe, verificamos salida; si no, al menos no debe ser "Bloqueado"
        if "WinError" in result or "Error de shell" in result:
            self.skipTest(f"cwd ~/Proyecto_V3 no existe en esta máquina: {result}")
        self.assertIn("hola", result)

    def test_shell_comando_bloqueado_rm(self):
        result = _tool_shell("rm -rf /")
        self.assertIn("Bloqueado", result)

    def test_shell_comando_bloqueado_sudo(self):
        result = _tool_shell("sudo ls")
        self.assertIn("Bloqueado", result)

    def test_shell_comando_bloqueado_pip_install(self):
        result = _tool_shell("pip install torch")
        self.assertIn("Bloqueado", result)

    def test_shell_comando_bloqueado_wget(self):
        result = _tool_shell("wget http://example.com")
        self.assertIn("Bloqueado", result)

    def test_shell_comando_bloqueado_curl(self):
        result = _tool_shell("curl http://example.com")
        self.assertIn("Bloqueado", result)

    def test_shell_comando_bloqueado_redireccion_dev(self):
        result = _tool_shell("echo data > /dev/sda")
        self.assertIn("Bloqueado", result)

    # ── http_get ────────────────────────────────────────────────────

    def test_http_get_solo_permite_http_https(self):
        result = _tool_http_get("ftp://example.com")
        self.assertIn("solo se permiten URLs", result)

    def test_http_get_file_protocolo_rechazado(self):
        result = _tool_http_get("file:///etc/passwd")
        self.assertIn("solo se permiten URLs", result)


# ===========================================================================
# I — AgentResult / AgentStep
# ===========================================================================

class TestAgentResult(unittest.TestCase):
    """Dataclasses de resultado."""

    def test_agent_step_to_dict(self):
        step = AgentStep(
            thought="Pensamiento de prueba",
            action="list_dir",
            action_input={"path": "."},
            observation="3 archivos encontrados",
        )
        d = step.to_dict()
        self.assertEqual(d["thought"], "Pensamiento de prueba")
        self.assertEqual(d["action"], "list_dir")
        self.assertEqual(d["observation"], "3 archivos encontrados")
        self.assertNotIn("final_answer", d)

    def test_agent_step_final_to_dict(self):
        step = AgentStep(
            thought="Todo listo",
            is_final=True,
            final_answer="Resultado final.",
        )
        d = step.to_dict()
        self.assertIn("final_answer", d)
        self.assertEqual(d["final_answer"], "Resultado final.")
        self.assertNotIn("action", d)

    def test_agent_result_to_dict(self):
        result = AgentResult(
            answer="Respuesta exitosa.",
            steps=[
                AgentStep(thought="Paso 1", action="list_dir", action_input={"path": "."}, observation="OK"),
                AgentStep(thought="Paso 2", is_final=True, final_answer="Respuesta exitosa."),
            ],
            success=True,
        )
        d = result.to_dict()
        self.assertEqual(d["answer"], "Respuesta exitosa.")
        self.assertEqual(d["steps_taken"], 2)
        self.assertTrue(d["success"])

    def test_agent_result_fallo_to_dict(self):
        result = AgentResult(
            answer="No se pudo completar.",
            steps=[],
            success=False,
        )
        d = result.to_dict()
        self.assertFalse(d["success"])
        self.assertEqual(d["steps_taken"], 0)


# ===========================================================================
# J — Tool (wrapper)
# ===========================================================================

class TestToolWrapper(unittest.TestCase):
    """Wrapper Tool: name, description, params_doc, __call__."""

    def _dummy_fn(self, nombre: str, edad: int = 0) -> str:
        return f"{nombre} tiene {edad} años"

    def test_tool_llamada_correcta(self):
        tool = Tool(
            name="saludar",
            description="Saluda a alguien",
            params_doc='{"nombre": "Juan", "edad": 30}',
            fn=self._dummy_fn,
        )
        result = tool(nombre="Ana", edad=25)
        self.assertEqual(result, "Ana tiene 25 años")

    def test_tool_error_parametros(self):
        tool = Tool(
            name="saludar",
            description="Saluda a alguien",
            params_doc='{"nombre": "Juan"}',
            fn=self._dummy_fn,
        )
        result = tool(parametro_incorrecto="valor")
        self.assertIn("Error de parámetros", result)

    def test_tool_error_ejecucion(self):
        def failing_fn(**kwargs):
            raise ValueError("fallo simulado")
        tool = Tool(
            name="falla",
            description="Siempre falla",
            params_doc="{}",
            fn=failing_fn,
        )
        result = tool()
        self.assertIn("Error en 'falla'", result)


# ===========================================================================
# K — DEFAULT_TOOLS
# ===========================================================================

class TestDefaultTools(unittest.TestCase):
    """Las 4 herramientas por defecto son válidas."""

    def test_cuatro_herramientas(self):
        self.assertEqual(len(DEFAULT_TOOLS), 4)

    def test_nombres_correctos(self):
        nombres = {t.name for t in DEFAULT_TOOLS}
        self.assertEqual(nombres, {"read_file", "list_dir", "shell", "http_get"})

    def test_todas_son_instancia_de_tool(self):
        for t in DEFAULT_TOOLS:
            self.assertIsInstance(t, Tool)

    def test_cada_tool_tiene_description_y_params_doc(self):
        for t in DEFAULT_TOOLS:
            self.assertTrue(t.description, f"{t.name} sin description")
            self.assertTrue(t.params_doc, f"{t.name} sin params_doc")


# ===========================================================================
# L — _SHELL_BLOCKED (regex de seguridad)
# ===========================================================================

class TestShellBlocked(unittest.TestCase):
    """Validación del regex de comandos bloqueados."""

    def test_rm_simple_bloqueado(self):
        self.assertTrue(_SHELL_BLOCKED.search("rm archivo.txt"))

    def test_rm_con_opciones_bloqueado(self):
        self.assertTrue(_SHELL_BLOCKED.search("rm -rf /tmp/datos"))

    def test_rm_en_pipeline_bloqueado(self):
        self.assertTrue(_SHELL_BLOCKED.search("find . -name '*.tmp' | xargs rm"))

    def test_dd_bloqueado(self):
        self.assertTrue(_SHELL_BLOCKED.search("dd if=/dev/zero of=archivo bs=1M count=100"))

    def test_mkfs_bloqueado(self):
        self.assertTrue(_SHELL_BLOCKED.search("mkfs.ext4 /dev/sda1"))

    def test_shutdown_bloqueado(self):
        self.assertTrue(_SHELL_BLOCKED.search("shutdown -h now"))

    def test_apt_bloqueado(self):
        self.assertTrue(_SHELL_BLOCKED.search("apt install python3"))

    def test_python_pip_install_bloqueado(self):
        self.assertTrue(_SHELL_BLOCKED.search("python -m pip install torch"))

    def test_ls_no_bloqueado(self):
        self.assertIsNone(_SHELL_BLOCKED.search("ls -la"))

    def test_cat_no_bloqueado(self):
        self.assertIsNone(_SHELL_BLOCKED.search("cat archivo.txt"))

    def test_grep_no_bloqueado(self):
        self.assertIsNone(_SHELL_BLOCKED.search("grep error archivo.log"))

    def test_find_no_bloqueado(self):
        self.assertIsNone(_SHELL_BLOCKED.search("find . -name '*.py'"))

    def test_nvidia_smi_no_bloqueado(self):
        self.assertIsNone(_SHELL_BLOCKED.search("nvidia-smi"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
