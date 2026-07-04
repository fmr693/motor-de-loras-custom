"""
tests/test_g5.py
================
G5 — from_docx() y from_html() en DataDigestor.

Cubre:
  A. from_docx: carga básica, filtro de párrafos cortos, FileNotFoundError,
     ImportError (mock), encadenado con to_jsonl(), from_folder() dispatch.
  B. from_html: extracción de body, selector CSS, sin contenido, FileNotFoundError,
     ImportError (mock), from_folder() dispatch.
  C. Integración: from_folder() para carpeta mixta .docx + .html.

Requiere python-docx y beautifulsoup4 (ambos disponibles en el entorno).
Si no estuvieran se marcan como skip automáticamente.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from motor.digestor import DataDigestor

# ---------------------------------------------------------------------------
# Marcadores de disponibilidad (pytest.importorskip actúa como guard)
# ---------------------------------------------------------------------------
docx_mod  = pytest.importorskip("docx",  reason="python-docx no instalado")
bs4_mod   = pytest.importorskip("bs4",   reason="beautifulsoup4 no instalado")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_docx(path: Path, paragraphs: list[str]) -> Path:
    """Crea un .docx mínimo con los párrafos dados."""
    from docx import Document
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))
    return path


def make_html(path: Path, body: str) -> Path:
    """Crea un .html mínimo con el body dado."""
    html = f"<html><body>{body}</body></html>"
    path.write_text(html, encoding="utf-8")
    return path


# ===========================================================================
# A. from_docx
# ===========================================================================

class TestFromDocx:

    def test_carga_parrafos(self, tmp_path):
        """Extrae párrafos válidos y genera ejemplos."""
        paragraphs = [
            "Este es el primer párrafo del documento de Word.",
            "Segundo párrafo con contenido técnico relevante.",
            "Tercer párrafo sobre inteligencia artificial y modelos de lenguaje.",
        ]
        f = make_docx(tmp_path / "test.docx", paragraphs)
        d = DataDigestor(task="general")
        d.from_docx(str(f))
        # Los 3 párrafos tienen >10 chars, todos deben aparecer
        assert len(d._examples) == 3

    def test_filtra_parrafos_cortos(self, tmp_path):
        """Párrafos con ≤10 caracteres son omitidos."""
        paragraphs = [
            "OK",  # corto
            "Párrafo largo suficiente para pasar el filtro de longitud.",
            "Otro párrafo suficientemente largo para ser incluido.",
        ]
        f = make_docx(tmp_path / "short.docx", paragraphs)
        d = DataDigestor(task="general")
        d.from_docx(str(f))
        assert len(d._examples) == 2

    def test_docx_vacio_no_crashea(self, tmp_path):
        """Un .docx sin párrafos no genera ejemplos pero tampoco falla."""
        f = make_docx(tmp_path / "empty.docx", [])
        d = DataDigestor(task="general")
        d.from_docx(str(f))
        assert len(d._examples) == 0

    def test_file_not_found(self, tmp_path):
        """from_docx lanza FileNotFoundError si el archivo no existe."""
        d = DataDigestor(task="general")
        with pytest.raises(FileNotFoundError):
            d.from_docx(str(tmp_path / "inexistente.docx"))

    def test_import_error_sin_docx(self, tmp_path):
        """Si python-docx no está instalado se lanza ImportError."""
        f = make_docx(tmp_path / "t.docx", ["Hola mundo, esto es un párrafo largo."])
        d = DataDigestor(task="general")
        # Ocultamos el módulo temporalmente
        with patch.dict(sys.modules, {"docx": None}):
            with pytest.raises((ImportError, TypeError)):
                d.from_docx(str(f))

    def test_encadenado_to_jsonl(self, tmp_path):
        """from_docx().to_jsonl() exporta los ejemplos al archivo."""
        paragraphs = [
            "Análisis de datos de entrenamiento con modelos grandes de lenguaje.",
            "Optimización de hiperparámetros en redes neuronales profundas.",
        ]
        docx_path = make_docx(tmp_path / "chain.docx", paragraphs)
        out_path  = tmp_path / "out.jsonl"
        d = DataDigestor(task="general")
        n = d.from_docx(str(docx_path)).to_jsonl(str(out_path), deduplicate=False)
        assert out_path.exists()
        assert n == 2
        lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_devuelve_self(self, tmp_path):
        """from_docx() devuelve self para encadenamiento."""
        f = make_docx(tmp_path / "r.docx", ["Párrafo para comprobar que se retorna self correctamente."])
        d = DataDigestor(task="general")
        result = d.from_docx(str(f))
        assert result is d

    def test_contenido_en_mensaje_user(self, tmp_path):
        """El texto del párrafo aparece en el campo user del mensaje."""
        texto = "Contenido especial del documento Word de prueba para test."
        f = make_docx(tmp_path / "content.docx", [texto])
        d = DataDigestor(task="general")
        d.from_docx(str(f))
        # Buscar el texto en los mensajes de los ejemplos
        all_content = " ".join(
            m["content"]
            for ex in d._examples
            for m in ex["messages"]
        )
        assert texto in all_content

    def test_from_folder_despacha_docx(self, tmp_path):
        """from_folder() procesa archivos .docx de la carpeta."""
        make_docx(
            tmp_path / "doc1.docx",
            ["Primer documento de la carpeta con párrafos suficientemente largos para ser incluidos."]
        )
        make_docx(
            tmp_path / "doc2.docx",
            ["Segundo documento de la carpeta con contenido de prueba para verificar el dispatch."]
        )
        d = DataDigestor(task="general")
        d.from_folder(str(tmp_path))
        assert len(d._examples) == 2


# ===========================================================================
# B. from_html
# ===========================================================================

class TestFromHtml:

    def test_extrae_texto_body(self, tmp_path):
        """Extrae bloques de texto del body correctamente."""
        html_content = (
            "<p>Este es el primer bloque de texto con suficiente longitud para ser incluido.</p>"
            "<p>Este es el segundo bloque de texto también suficientemente largo para pasar.</p>"
        )
        f = make_html(tmp_path / "test.html", html_content)
        d = DataDigestor(task="general")
        d.from_html(str(f))
        assert len(d._examples) >= 1

    def test_ignora_scripts_y_estilos(self, tmp_path):
        """El contenido de <script> y <style> no se incluye en los ejemplos."""
        html_content = (
            "<script>var x = 42; function secret() { return true; }</script>"
            "<style>.hidden { display: none; }</style>"
            "<p>Texto visible del artículo que debe ser extraido correctamente.</p>"
        )
        f = make_html(tmp_path / "scripts.html", html_content)
        d = DataDigestor(task="general")
        d.from_html(str(f))
        all_content = " ".join(
            m["content"] for ex in d._examples for m in ex["messages"]
        )
        assert "var x = 42" not in all_content
        assert ".hidden" not in all_content

    def test_selector_css(self, tmp_path):
        """text_selector limita la extracción a un contenedor CSS."""
        html_content = (
            "<div class='sidebar'>Sidebar con texto que no debería ser incluido.</div>"
            "<article><p>Contenido principal del artículo que debe ser extraido por el selector CSS.</p></article>"
        )
        f = make_html(tmp_path / "selector.html", html_content)
        d = DataDigestor(task="general")
        d.from_html(str(f), text_selector="article")
        all_content = " ".join(
            m["content"] for ex in d._examples for m in ex["messages"]
        )
        assert "Contenido principal" in all_content

    def test_selector_inexistente_usa_body(self, tmp_path):
        """Si el selector CSS no existe, hace fallback al body completo."""
        html_content = (
            "<p>Texto de prueba suficientemente largo para ser considerado un bloque válido.</p>"
        )
        f = make_html(tmp_path / "fallback.html", html_content)
        d = DataDigestor(task="general")
        # Selector que no existe
        d.from_html(str(f), text_selector=".no-existe-este-selector")
        # Debe funcionar y extraer del body
        assert len(d._examples) >= 1

    def test_html_sin_cuerpo_no_crashea(self, tmp_path):
        """Un HTML vacío o sin body no genera ejemplos pero no falla."""
        f = tmp_path / "empty.html"
        f.write_text("<html></html>", encoding="utf-8")
        d = DataDigestor(task="general")
        d.from_html(str(f))
        assert len(d._examples) == 0

    def test_file_not_found(self, tmp_path):
        """from_html lanza FileNotFoundError si el archivo no existe."""
        d = DataDigestor(task="general")
        with pytest.raises(FileNotFoundError):
            d.from_html(str(tmp_path / "no_existe.html"))

    def test_import_error_sin_bs4(self, tmp_path):
        """Si beautifulsoup4 no está instalado se lanza ImportError."""
        f = make_html(tmp_path / "t.html", "<p>Texto de prueba suficientemente largo para el test.</p>")
        d = DataDigestor(task="general")
        with patch.dict(sys.modules, {"bs4": None}):
            with pytest.raises((ImportError, TypeError)):
                d.from_html(str(f))

    def test_devuelve_self(self, tmp_path):
        """from_html() devuelve self para encadenamiento."""
        f = make_html(tmp_path / "r.html",
                      "<p>Texto de prueba suficientemente largo para el test de encadenamiento.</p>")
        d = DataDigestor(task="general")
        result = d.from_html(str(f))
        assert result is d

    def test_encadenado_to_jsonl(self, tmp_path):
        """from_html().to_jsonl() exporta los ejemplos."""
        html_content = (
            "<p>Bloque largo de texto HTML para exportar a JSONL con DataDigestor sin problemas.</p>"
        )
        html_path = make_html(tmp_path / "chain.html", html_content)
        out_path  = tmp_path / "out_html.jsonl"
        d = DataDigestor(task="general")
        n = d.from_html(str(html_path)).to_jsonl(str(out_path), deduplicate=False)
        assert out_path.exists()
        assert n >= 1

    def test_from_folder_despacha_html(self, tmp_path):
        """from_folder() procesa archivos .html y .htm de la carpeta."""
        make_html(
            tmp_path / "page.html",
            "<p>Primer archivo HTML de la carpeta suficientemente largo para pasar el filtro.</p>",
        )
        make_html(
            tmp_path / "otra.htm",
            "<p>Segundo archivo HTM de la carpeta suficientemente largo para pasar el filtro.</p>",
        )
        d = DataDigestor(task="general")
        d.from_folder(str(tmp_path))
        assert len(d._examples) >= 2


# ===========================================================================
# C. Integración: carpeta mixta .docx + .html
# ===========================================================================

class TestFromFolderMixta:

    def test_carpeta_mixta_docx_y_html(self, tmp_path):
        """from_folder() procesa a la vez .docx y .html en la misma carpeta."""
        make_docx(
            tmp_path / "informe.docx",
            ["Informe trimestral con análisis detallado de las métricas del negocio para el equipo."]
        )
        make_html(
            tmp_path / "resumen.html",
            "<article><p>Resumen ejecutivo del trimestre con datos clave para la presentación directiva.</p></article>",
        )
        d = DataDigestor(task="general")
        d.from_folder(str(tmp_path))
        assert len(d._examples) == 2

    def test_carpeta_solo_docx(self, tmp_path):
        """from_folder() con solo .docx funciona sin HTML."""
        make_docx(
            tmp_path / "doc.docx",
            ["Documento solo de Word con contenido suficientemente largo para ser incluido."]
        )
        d = DataDigestor(task="general")
        d.from_folder(str(tmp_path))
        assert len(d._examples) == 1

    def test_carpeta_solo_html(self, tmp_path):
        """from_folder() con solo .html funciona sin DOCX."""
        make_html(
            tmp_path / "page.html",
            "<p>Página HTML con suficiente texto para ser reconocida como bloque válido de contenido.</p>"
        )
        d = DataDigestor(task="general")
        d.from_folder(str(tmp_path))
        assert len(d._examples) >= 1
