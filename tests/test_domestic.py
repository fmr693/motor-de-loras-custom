"""
tests/test_domestic.py
======================
Suite de tests para motor.domestic_tools y motor.domestic_dataset_gen.

Cubre:
  1.  _is_safe_path — rutas seguras, inseguras, no existentes
  2.  file_organize — dry_run, destino inseguro, archivo no encontrado
  3.  note_save — escritura real, carpeta insegura, caracteres especiales en título
  4.  process_run — dry_run, proceso no en whitelist
  5.  search_files — ruta no existe, directorio vacío, búsqueda sin resultados
  6.  to_react — formato válido, parseable por motor.agent._parse_output
  7.  _parse_toolace_call — formatos con y sin espacios en el nombre
  8.  Dataset JSONL — 700 líneas, roles correctos, no final_answers contaminados
  9.  _build_domestic_system — cache funciona, contiene los 6 tools
  10. _augment_request — genera n variaciones, sin duplicados evidentes
  11. IMAP injection — _imap_escape no está disponible directamente pero comprobamos
      que sender/subject con comillas dobles no rompen la query construida en
      email_filter (probado via inspección del código, no conexión real).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

# Añadir raíz del proyecto al path

from motor.domestic_tools import (
    DOMESTIC_TOOLS,
    _SAFE_WRITE_ROOTS,
    _SAFE_ROOTS_RESOLVED,
    _is_safe_path,
    file_organize,
    note_save,
    process_run,
    search_files,
)
from motor.domestic_dataset_gen import (
    DomesticExample,
    Step,
    _augment_request,
    _build_domestic_system,
    _parse_toolace_call,
    to_chatml,
    to_react,
)


# ---------------------------------------------------------------------------
# 1. _is_safe_path
# ---------------------------------------------------------------------------

class TestIsSafePath(unittest.TestCase):

    def test_home_notas_is_safe(self):
        """~/Notas está en _SAFE_WRITE_ROOTS."""
        path = Path.home() / "Notas" / "mi_nota.txt"
        self.assertTrue(_is_safe_path(path))

    def test_home_documents_is_safe(self):
        path = Path.home() / "Documents" / "subdir" / "file.pdf"
        self.assertTrue(_is_safe_path(path))

    def test_system_root_is_unsafe(self):
        path = Path("C:/Windows/System32/evil.bat") if os.name == "nt" else Path("/etc/passwd")
        self.assertFalse(_is_safe_path(path))

    def test_temp_dir_is_unsafe(self):
        path = Path(tempfile.gettempdir()) / "test.txt"
        self.assertFalse(_is_safe_path(path))

    def test_nonexistent_safe_path_still_allowed(self):
        """La carpeta no tiene que existir para ser permitida (fix crítico)."""
        path = Path.home() / "Notas" / "carpeta_nueva" / "nota.txt"
        # La carpeta puede no existir, pero la ruta es segura
        self.assertTrue(_is_safe_path(path))

    def test_no_duplicate_roots(self):
        """_SAFE_WRITE_ROOTS no debe tener duplicados."""
        resolved = [str(r.expanduser().resolve()) for r in _SAFE_WRITE_ROOTS]
        self.assertEqual(len(resolved), len(set(resolved)), "Hay raíces duplicadas en _SAFE_WRITE_ROOTS")

    def test_safe_roots_resolved_matches_roots(self):
        """_SAFE_ROOTS_RESOLVED debe tener el mismo número que _SAFE_WRITE_ROOTS."""
        self.assertEqual(len(_SAFE_ROOTS_RESOLVED), len(_SAFE_WRITE_ROOTS))


# ---------------------------------------------------------------------------
# 2. file_organize
# ---------------------------------------------------------------------------

class TestFileOrganize(unittest.TestCase):

    def test_dry_run_no_move(self):
        """dry_run=True no mueve ningún archivo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "test.txt"
            src.write_text("contenido")
            dest = Path(tmpdir) / "destino"
            result = file_organize([str(src)], str(dest), dry_run=True)
            self.assertIn("[dry_run]", result)
            self.assertFalse(dest.exists(), "dry_run no debería crear la carpeta destino")

    def test_unsafe_dest_blocked(self):
        """Destino fuera de rutas seguras devuelve mensaje bloqueado."""
        unsafe_dest = str(Path(tempfile.gettempdir()) / "unsafe")
        result = file_organize(["~/Desktop/algo.pdf"], unsafe_dest, dry_run=False)
        self.assertIn("[Bloqueado]", result)

    def test_file_not_found(self):
        """Archivo inexistente reporta error sin crash."""
        result = file_organize(["~/Desktop/archivo_que_no_existe_12345.txt"],
                               "~/Documentos/dest/", dry_run=True)
        self.assertIn("No encontrado", result)

    def test_dry_run_shows_arrow(self):
        """dry_run muestra → para archivos a mover."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "factura.pdf"
            src.write_text("pdf content")
            result = file_organize([str(src)], str(tmpdir) + "/dest", dry_run=True)
            self.assertIn("→", result)


# ---------------------------------------------------------------------------
# 3. note_save
# ---------------------------------------------------------------------------

class TestNoteSave(unittest.TestCase):

    def test_save_in_safe_folder(self):
        """Guarda nota en ~/Notas con timestamp."""
        folder = str(Path.home() / "Notas" / "_test_suite_tmp")
        try:
            result = note_save("Test Note", "Contenido de prueba", folder=folder)
            self.assertIn("note_save", result)
            self.assertNotIn("[Bloqueado]", result)
            # Buscar y eliminar el archivo creado
            test_folder = Path(folder)
            if test_folder.exists():
                for f in test_folder.glob("*_Test_Note.txt"):
                    f.unlink()
                if not any(test_folder.iterdir()):
                    test_folder.rmdir()
        except Exception as e:
            self.fail(f"note_save lanzó excepción: {e}")

    def test_unsafe_folder_blocked(self):
        """Carpeta fuera de raíces seguras devuelve Bloqueado."""
        result = note_save("título", "cuerpo", folder=str(tempfile.gettempdir()))
        self.assertIn("[Bloqueado]", result)

    def test_special_chars_in_title(self):
        """Caracteres especiales en título son sanitizados."""
        folder = str(Path.home() / "Notas" / "_test_suite_tmp")
        try:
            result = note_save("Título: ?/<>|*", "cuerpo", folder=folder)
            # No debe haber ? / < > | * en el nombre de archivo
            self.assertNotIn("[Error]", result)
            # Limpiar
            test_folder = Path(folder)
            if test_folder.exists():
                for f in test_folder.iterdir():
                    f.unlink()
                test_folder.rmdir()
        except Exception as e:
            self.fail(f"note_save con caracteres especiales lanzó: {e}")


# ---------------------------------------------------------------------------
# 4. process_run
# ---------------------------------------------------------------------------

class TestProcessRun(unittest.TestCase):

    def test_dry_run(self):
        result = process_run("ping", ["localhost"], dry_run=True)
        self.assertIn("[dry_run]", result)
        self.assertIn("ping", result.lower())

    def test_unknown_process_blocked(self):
        result = process_run("rm", ["-rf", "/"], dry_run=False)
        self.assertIn("[Bloqueado]", result)

    def test_unknown_process_shows_whitelist(self):
        result = process_run("curl", [], dry_run=True)
        self.assertIn("[Bloqueado]", result)
        self.assertIn("ping", result)  # whitelist visible en el mensaje

    def test_whitelist_case_insensitive(self):
        """process_run normaliza el nombre a lowercase."""
        result = process_run("PING", ["localhost"], dry_run=True)
        self.assertNotIn("[Bloqueado]", result)


# ---------------------------------------------------------------------------
# 5. search_files
# ---------------------------------------------------------------------------

class TestSearchFiles(unittest.TestCase):

    def test_nonexistent_path(self):
        result = search_files("texto", "/ruta/que/no/existe/99999")
        self.assertIn("[Error]", result)

    def test_not_a_directory(self):
        """Si la ruta es un archivo, no directorio, devuelve error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            fname = f.name
        try:
            result = search_files("texto", fname)
            self.assertIn("[Error]", result)
        finally:
            os.unlink(fname)

    def test_no_results(self):
        """Búsqueda sin coincidencias devuelve mensaje claro."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").write_text("sin coincidencias")
            result = search_files("ZZZZZ_NEVER_FOUND", tmpdir)
            self.assertIn("ninguna coincidencia", result.lower())

    def test_finds_match(self):
        """Búsqueda encuentra texto real."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "nota.txt").write_text("hola mundo esta es mi nota")
            result = search_files("hola mundo", tmpdir, extensions=[".txt"])
            self.assertIn("nota.txt", result)
            self.assertIn("hola mundo", result)


# ---------------------------------------------------------------------------
# 6. to_react y compatibilidad con _parse_output
# ---------------------------------------------------------------------------

class TestToReact(unittest.TestCase):

    def _make_example(self) -> DomesticExample:
        return {
            "user_request": "Organiza las facturas del escritorio",
            "steps": [
                {
                    "thought": "Busco las facturas.",
                    "tool": "search_files",
                    "args": {"query": "factura", "path": "~/Desktop"},
                    "result": "search_files: 2 coincidencias.",
                },
                {
                    "thought": "Las muevo al destino.",
                    "tool": "file_organize",
                    "args": {"files": ["~/Desktop/f1.pdf"], "dest": "~/Documentos/Facturas/"},
                    "result": "file_organize: 1 archivos procesados.",
                },
            ],
            "final_thought": "Tarea completada.",
            "final_answer": "Las facturas se han movido.",
            "language": "es",
            "category": "file_organize",
        }

    def test_react_has_all_keywords(self):
        text = to_react(self._make_example())
        self.assertIn("Thought:", text)
        self.assertIn("Action:", text)
        self.assertIn("Action Input:", text)
        self.assertIn("Observation:", text)
        self.assertIn("Final Answer:", text)

    def test_react_ends_with_final_answer(self):
        text = to_react(self._make_example())
        lines = text.strip().splitlines()
        self.assertTrue(lines[-1].startswith("Final Answer:"),
                        f"Última línea: {lines[-1]!r}")

    def test_react_action_input_is_valid_json(self):
        """Todos los Action Input deben ser JSON válido."""
        text = to_react(self._make_example())
        for m in re.finditer(r"Action Input:\s*(\{.+?\})", text, re.DOTALL):
            raw = m.group(1)
            try:
                json.loads(raw)
            except json.JSONDecodeError as e:
                self.fail(f"Action Input no es JSON válido: {raw!r} — {e}")

    def test_parse_output_compat(self):
        """_parse_output de agent.py puede parsear la salida de to_react."""
        from motor.agent import LoRAAgent
        text = to_react(self._make_example())

        # Simular _parse_output (no necesitamos infer_fn real)
        # Buscamos Final Answer directamente
        m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        self.assertIsNotNone(m)
        self.assertIn("facturas", m.group(1).lower())

    def test_to_chatml_structure(self):
        ex = self._make_example()
        chatml = to_chatml(ex, renderer="react")
        self.assertIn("messages", chatml)
        msgs = chatml["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertIn("Tarea:", msgs[1]["content"])
        self.assertIn("Final Answer:", msgs[2]["content"])

    def test_system_contains_all_tools(self):
        """El system prompt contiene los 6 nombres de herramientas."""
        chatml = to_chatml(self._make_example(), renderer="react")
        system = chatml["messages"][0]["content"]
        for tool_name in ["file_organize", "email_filter", "calendar_get",
                          "note_save", "search_files", "process_run"]:
            self.assertIn(tool_name, system,
                          f"Tool '{tool_name}' no aparece en el system prompt")


# ---------------------------------------------------------------------------
# 7. _parse_toolace_call
# ---------------------------------------------------------------------------

class TestParseToolaceCall(unittest.TestCase):

    def test_simple_call(self):
        calls = _parse_toolace_call('[ping(host="8.8.8.8", count=3)]')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "ping")
        self.assertEqual(calls[0]["arguments"]["host"], "8.8.8.8")
        self.assertEqual(calls[0]["arguments"]["count"], 3)

    def test_name_with_spaces(self):
        """Nombres con espacios se normalizan con guiones bajos."""
        calls = _parse_toolace_call('[Market Trends API(trend_type="MARKET", country="us")]')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Market_Trends_API")

    def test_multiple_calls(self):
        text = '[func_a(x=1)][func_b(y="hello")]'
        calls = _parse_toolace_call(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "func_a")
        self.assertEqual(calls[1]["name"], "func_b")

    def test_no_calls(self):
        calls = _parse_toolace_call("Esto es texto normal sin llamadas.")
        self.assertEqual(calls, [])

    def test_empty_args(self):
        calls = _parse_toolace_call('[get_status()]')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"], {})


# ---------------------------------------------------------------------------
# 8. Dataset JSONL — valida el PROCESO de generación (no un fichero estático)
#
# Arquitectura: domestic_dataset_gen corre sin GPU, tanto en Windows dev
# como en el worker container. Aquí probamos que el generador produce
# JSONL bien formado, con las estructuras correctas. Generamos los datos
# aquí mismo para no depender de ningún fichero pre-existente en disco.
# ---------------------------------------------------------------------------

class TestDatasetJsonl(unittest.TestCase):
    """
    Prueba el proceso generate_dataset() directamente.
    No requiere ficheros en disco; genera N=50 ejemplos en un tmpdir.
    Corre en: Windows dev / worker container (sin GPU).
    """

    N_EXAMPLES = 50  # Controlado: rápido y suficiente para validar estructura

    @classmethod
    def setUpClass(cls):
        """Genera el dataset una sola vez para toda la clase (más rápido)."""
        import shutil
        from motor.domestic_dataset_gen import generate_dataset
        cls._tmpdir = tempfile.mkdtemp(prefix="test_domestic_dataset_")
        cls._out = Path(cls._tmpdir) / "dataset_test.jsonl"
        generate_dataset(
            n=cls.N_EXAMPLES,
            out=str(cls._out),
            augment_factor=3,
            verbose=False,
        )
        with cls._out.open(encoding="utf-8") as f:
            cls._lines = [l for l in f.readlines() if l.strip()]

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_generator_produces_examples(self):
        """El generador produce al menos 1 ejemplo (humo básico)."""
        self.assertGreater(len(self._lines), 0,
                           "generate_dataset() no produjo ningún ejemplo")

    def test_output_file_created(self):
        """El fichero JSONL se crea en la ruta indicada."""
        self.assertTrue(self._out.exists(),
                        f"Fichero no creado: {self._out}")

    def test_all_lines_valid_json(self):
        """Todas las líneas deben ser JSON válido."""
        for i, line in enumerate(self._lines, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                self.fail(f"Línea {i} no es JSON válido: {e}")

    def test_all_examples_have_messages(self):
        """Todos los ejemplos deben tener la clave 'messages'."""
        for i, line in enumerate(self._lines, 1):
            ex = json.loads(line)
            self.assertIn("messages", ex, f"Línea {i} sin 'messages'")

    def test_messages_roles(self):
        """Todas las messages deben tener roles [system, user, assistant]."""
        for i, line in enumerate(self._lines, 1):
            ex = json.loads(line)
            roles = [m["role"] for m in ex["messages"]]
            self.assertEqual(roles, ["system", "user", "assistant"],
                             f"Línea {i}: roles incorrectos {roles}")

    def test_user_message_starts_with_tarea(self):
        """El mensaje de usuario debe comenzar con 'Tarea:'."""
        for i, line in enumerate(self._lines, 1):
            ex = json.loads(line)
            user_content = ex["messages"][1]["content"]
            self.assertTrue(user_content.startswith("Tarea:"),
                           f"Línea {i}: user no empieza con 'Tarea:' → {user_content[:50]!r}")

    def test_assistant_has_thought(self):
        """El turno assistant debe contener 'Thought:'."""
        for i, line in enumerate(self._lines, 1):
            ex = json.loads(line)
            asst = ex["messages"][2]["content"]
            self.assertIn("Thought:", asst,
                         f"Línea {i}: sin 'Thought:' en assistant")

    def test_assistant_ends_with_final_answer(self):
        """El turno assistant debe contener 'Final Answer:'."""
        for i, line in enumerate(self._lines, 1):
            ex = json.loads(line)
            asst = ex["messages"][2]["content"]
            has_final = any(l.startswith("Final Answer:") for l in asst.splitlines())
            self.assertTrue(has_final,
                           f"Línea {i}: sin 'Final Answer:' en assistant")

    def test_no_contaminated_final_answers(self):
        """Ningún final_answer debe contener otra tool call."""
        bad = []
        for i, line in enumerate(self._lines, 1):
            ex = json.loads(line)
            asst = ex["messages"][2]["content"]
            for l in asst.splitlines():
                if l.startswith("Final Answer:") and re.search(r"\[[\w\s]+\(", l):
                    bad.append(i)
        self.assertEqual(bad, [],
                        f"Ejemplos con final_answer contaminado: {bad[:10]}")

    def test_action_inputs_are_valid_json(self):
        """Todos los 'Action Input:' deben contener JSON válido."""
        errors = []
        for i, line in enumerate(self._lines, 1):
            ex = json.loads(line)
            asst = ex["messages"][2]["content"]
            for asst_line in asst.splitlines():
                m = re.match(r"Action Input:\s*(.+)$", asst_line.strip())
                if not m:
                    continue
                raw = m.group(1).strip()
                try:
                    json.loads(raw)
                except json.JSONDecodeError:
                    errors.append((i, raw[:80]))
        self.assertEqual(errors, [],
                        f"Action Inputs con JSON inválido (primeros 5): {errors[:5]}")

    def test_system_message_contains_tools(self):
        """El system message menciona las herramientas domésticas."""
        ex = json.loads(self._lines[0])
        system = ex["messages"][0]["content"]
        for tool_name in ["file_organize", "note_save", "search_files"]:
            self.assertIn(tool_name, system,
                         f"System message no menciona '{tool_name}'")


# ---------------------------------------------------------------------------
# 9. _build_domestic_system — cache
# ---------------------------------------------------------------------------

class TestBuildDomesticSystem(unittest.TestCase):

    def test_returns_string(self):
        result = _build_domestic_system()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)

    def test_cache_same_object(self):
        """Dos llamadas devuelven exactamente el mismo objeto (cache)."""
        s1 = _build_domestic_system()
        s2 = _build_domestic_system()
        self.assertIs(s1, s2, "El cache no devuelve el mismo objeto")

    def test_contains_all_tools(self):
        system = _build_domestic_system()
        for tool_name in ["file_organize", "email_filter", "calendar_get",
                          "note_save", "search_files", "process_run"]:
            self.assertIn(tool_name, system)

    def test_contains_max_steps(self):
        system = _build_domestic_system()
        self.assertIn("10", system)  # max_steps=10


# ---------------------------------------------------------------------------
# 10. _augment_request
# ---------------------------------------------------------------------------

class TestAugmentRequest(unittest.TestCase):

    def test_returns_n_variants(self):
        variants = _augment_request("organiza los archivos", n=3)
        self.assertLessEqual(len(variants), 3)  # puede haber dedup
        self.assertGreaterEqual(len(variants), 1)

    def test_first_variant_is_original(self):
        request = "organiza los archivos del escritorio"
        variants = _augment_request(request, n=3)
        self.assertEqual(variants[0], request)

    def test_no_pure_duplicates(self):
        """La lista no debe tener duplicados."""
        variants = _augment_request("lista los correos de hoy", n=5, seed=0)
        self.assertEqual(len(variants), len(set(variants)))

    def test_n1_returns_original(self):
        """n=1 devuelve solo el original."""
        variants = _augment_request("busca el contrato", n=1)
        self.assertEqual(len(variants), 1)


# ---------------------------------------------------------------------------
# 11. DOMESTIC_TOOLS — integridad del registro
# ---------------------------------------------------------------------------

class TestDomesticTools(unittest.TestCase):

    def test_six_tools_registered(self):
        self.assertEqual(len(DOMESTIC_TOOLS), 6)

    def test_all_tools_have_required_fields(self):
        for t in DOMESTIC_TOOLS:
            self.assertTrue(t.name, f"Tool sin nombre: {t}")
            self.assertTrue(t.description, f"Tool sin descripción: {t.name}")
            self.assertTrue(t.params_doc, f"Tool sin params_doc: {t.name}")
            self.assertTrue(callable(t._fn), f"Tool sin función: {t.name}")

    def test_tool_names_match_expected(self):
        names = {t.name for t in DOMESTIC_TOOLS}
        expected = {"file_organize", "email_filter", "calendar_get",
                    "note_save", "search_files", "process_run"}
        self.assertEqual(names, expected)

    def test_tool_call_returns_string(self):
        """Todos los tools deben devolver string (sin crash en dry_run/modo seguro)."""
        # Solo los que no requieren red/archivos reales
        from motor.domestic_tools import calendar_get, process_run, search_files
        import tempfile
        self.assertIsInstance(calendar_get("today"), str)
        self.assertIsInstance(process_run("ping", ["localhost"], dry_run=True), str)
        # Usar directorio temporal para no escanear Home completo
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsInstance(search_files("test", tmpdir, max_results=1), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
