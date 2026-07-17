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

    def test_pdf_extrae_con_libreria(self, tmp_path, monkeypatch):
        # con pypdf disponible (mockeado), el PDF se digiere como cualquier doc
        monkeypatch.setattr(_dig, "_pdf_to_text",
                            lambda p: "# Contrato\nRescisión con 30 días de aviso.")
        pdf = tmp_path / "contrato.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        d = DataDigestor(mode="knowledge", auto_enrich=False)
        d.from_document_knowledge(pdf, level="template")
        ex = d.get_examples()
        assert ex and "30 días" in ex[0]["messages"][2]["content"]

    def test_pdf_sin_libreria_degrada_con_aviso(self, tmp_path, monkeypatch, capsys):
        # sin pypdf: se OMITE con aviso claro, nunca rompe (Opción A)
        def _boom(p):
            raise ImportError("No module named 'pypdf'")
        monkeypatch.setattr(_dig, "_pdf_to_text", _boom)
        pdf = tmp_path / "contrato.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        d = DataDigestor(mode="knowledge", auto_enrich=False)
        d.from_document_knowledge(pdf, level="template")
        assert d.get_examples() == []                 # no rompió, no produjo
        out = capsys.readouterr().out
        assert "pypdf" in out and "AVISO" in out       # avisó qué falta

    def test_carpeta_mixta_md_ok_pdf_sin_lib(self, tmp_path, monkeypatch):
        # .md real se procesa; el .pdf sin librería se omite (dataset no vacío)
        monkeypatch.setattr(_dig, "_pdf_to_text",
                            lambda p: (_ for _ in ()).throw(ImportError("x")))
        (tmp_path / "notas.md").write_text("# Tema\nContenido de dominio real.",
                                           encoding="utf-8")
        (tmp_path / "extra.pdf").write_bytes(b"%PDF-1.4 fake")
        d = DataDigestor(mode="knowledge", auto_enrich=False)
        d.from_document_knowledge(tmp_path, level="template")
        assert d.get_examples()                        # el .md sí produjo


# ---------------------------------------------------------------------------
# F5 — Exposición de distill/knowledge en el CLI (digestor --mode)
# ---------------------------------------------------------------------------

import json as _json


def _run_cli(argv):
    """Construye el parser real de fabrica_loras y ejecuta el subcomando."""
    from fabrica_loras import _build_parser
    args = _build_parser().parse_args(argv)
    args.func(args)


def _rows(path):
    return [_json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


class TestCLIModos:
    def test_classify_sin_task_falla(self, tmp_path):
        # El modo por defecto es classify → sin --task debe abortar con aviso.
        data = tmp_path / "d.csv"
        data.write_text("text,label\nhola,1\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"
        with pytest.raises(SystemExit):
            _run_cli(["digestor", "--data", str(data), "--output", str(out)])

    def test_default_mode_es_classify(self, tmp_path):
        # Invocación clásica (sin --mode) sigue funcionando: retrocompat EXIST.
        data = tmp_path / "d.csv"
        data.write_text("text,label\nbueno,POS\nmalo,NEG\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"
        _run_cli(["digestor", "--data", str(data), "--label-col", "label",
                  "--task", "Clasifica el sentimiento.", "--output", str(out)])
        assert _rows(out)

    def test_distill_e2e(self, tmp_path):
        md = tmp_path / "chat.md"
        md.write_text("## Human\n¿Cuánto es 2+2?\n## Assistant\nSoy Claude. Son 4.",
                      encoding="utf-8")
        out = tmp_path / "sft.jsonl"
        _run_cli(["digestor", "--mode", "distill",
                  "--data", str(md), "--output", str(out)])
        rows = _rows(out)
        assert rows
        msgs = rows[0]["messages"]
        assert msgs[0]["role"] == "system"
        joined = " ".join(m["content"] for m in msgs)
        assert "Soy Claude" not in joined          # higiene por defecto
        assert "Son 4." in joined                  # el dato útil sobrevive

    def test_distill_keep_identity_opt_out(self, tmp_path):
        md = tmp_path / "chat.md"
        md.write_text("## Human\nhola\n## Assistant\nSoy Claude, un asistente. Hola de nuevo.",
                      encoding="utf-8")
        out = tmp_path / "sft.jsonl"
        _run_cli(["digestor", "--mode", "distill", "--keep-identity",
                  "--data", str(md), "--output", str(out)])
        joined = " ".join(m["content"] for m in _rows(out)[0]["messages"])
        assert "Soy Claude" in joined              # opt-out respetado

    def test_knowledge_template_e2e(self, tmp_path):
        doc = tmp_path / "manual.md"
        doc.write_text("# Rescisión\nEl contrato se rescinde con 30 días de aviso.",
                       encoding="utf-8")
        out = tmp_path / "qa.jsonl"
        _run_cli(["digestor", "--mode", "knowledge", "--level", "template",
                  "--data", str(doc), "--output", str(out)])
        rows = _rows(out)
        assert rows
        assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]

    def test_knowledge_completion_e2e(self, tmp_path):
        doc = tmp_path / "notas.md"
        doc.write_text("Primer bloque de dominio.\n\nSegundo bloque distinto.",
                       encoding="utf-8")
        out = tmp_path / "raw.jsonl"
        _run_cli(["digestor", "--mode", "knowledge", "--level", "completion",
                  "--chunk-chars", "25", "--data", str(doc), "--output", str(out)])
        rows = _rows(out)
        assert rows and all("text" in r for r in rows)   # continued-pretraining


# ---------------------------------------------------------------------------
# F5 — Router de modo (detect_digest_mode) + CLI --mode auto
# ---------------------------------------------------------------------------

_DIALOGO = "## Human\n¿Qué es X?\n## Assistant\nX es algo concreto y útil."
_DOCUMENTO = "# Título\nEsto es prosa expositiva sobre un dominio, sin diálogo."


class TestRouterDetectMode:
    def test_carpeta_de_charlas_es_distill(self, tmp_path):
        (tmp_path / "a.md").write_text(_DIALOGO, encoding="utf-8")
        (tmp_path / "b.md").write_text(_DIALOGO, encoding="utf-8")
        assert _dig.detect_digest_mode(tmp_path) == "distill"

    def test_documento_es_knowledge(self, tmp_path):
        (tmp_path / "manual.md").write_text(_DOCUMENTO, encoding="utf-8")
        assert _dig.detect_digest_mode(tmp_path) == "knowledge"

    def test_tabular_es_classify(self, tmp_path):
        (tmp_path / "datos.csv").write_text("text,label\nx,1\n", encoding="utf-8")
        assert _dig.detect_digest_mode(tmp_path) == "classify"

    def test_imagenes_es_vlm(self, tmp_path):
        (tmp_path / "img1.png").write_bytes(b"\x89PNG\r\n")
        (tmp_path / "img2.jpg").write_bytes(b"\xff\xd8\xff")
        assert _dig.detect_digest_mode(tmp_path) == "vlm"

    def test_archivo_unico_diálogo(self, tmp_path):
        f = tmp_path / "chat.md"
        f.write_text(_DIALOGO, encoding="utf-8")
        assert _dig.detect_digest_mode(f) == "distill"

    def test_mezcla_charla_y_documento_falla(self, tmp_path):
        (tmp_path / "chat.md").write_text(_DIALOGO, encoding="utf-8")
        (tmp_path / "doc.md").write_text(_DOCUMENTO, encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            _dig.detect_digest_mode(tmp_path)
        # el aviso menciona ambos modos en conflicto
        assert "distill" in str(exc.value) and "knowledge" in str(exc.value)

    def test_mezcla_texto_e_imagen_falla(self, tmp_path):
        (tmp_path / "doc.md").write_text(_DOCUMENTO, encoding="utf-8")
        (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n")
        with pytest.raises(ValueError):
            _dig.detect_digest_mode(tmp_path)

    def test_carpeta_vacia_falla(self, tmp_path):
        with pytest.raises(ValueError):
            _dig.detect_digest_mode(tmp_path)

    def test_ficheros_irrelevantes_se_ignoran(self, tmp_path):
        # un .DS_Store junto a charlas no rompe la detección
        (tmp_path / "chat.md").write_text(_DIALOGO, encoding="utf-8")
        (tmp_path / ".DS_Store").write_bytes(b"\x00\x01")
        assert _dig.detect_digest_mode(tmp_path) == "distill"

    def test_pdf_folder_es_knowledge(self, tmp_path):
        # PDF/DOCX se enrutan a knowledge (que los extrae o degrada con aviso)
        (tmp_path / "informe.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "notas.docx").write_bytes(b"PK fake")
        assert _dig.detect_digest_mode(tmp_path) == "knowledge"

    def test_charla_mas_pdf_es_mezcla(self, tmp_path):
        # charla (distill) + documento pdf (knowledge) → ambiguo → error claro
        (tmp_path / "chat.md").write_text(_DIALOGO, encoding="utf-8")
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
        with pytest.raises(ValueError) as exc:
            _dig.detect_digest_mode(tmp_path)
        assert "distill" in str(exc.value) and "knowledge" in str(exc.value)


class TestCLIAuto:
    def test_auto_distill_e2e(self, tmp_path):
        (tmp_path / "chat.md").write_text(_DIALOGO, encoding="utf-8")
        out = tmp_path / "out.jsonl"
        _run_cli(["digestor", "--mode", "auto",
                  "--data", str(tmp_path), "--output", str(out)])
        rows = _rows(out)
        assert rows and "messages" in rows[0]

    def test_auto_knowledge_e2e(self, tmp_path):
        (tmp_path / "manual.md").write_text(_DOCUMENTO, encoding="utf-8")
        out = tmp_path / "out.jsonl"
        _run_cli(["digestor", "--mode", "auto",
                  "--data", str(tmp_path), "--output", str(out)])
        rows = _rows(out)
        assert rows and rows[0]["messages"][0]["role"] == "system"

    def test_auto_mezclado_aborta(self, tmp_path):
        (tmp_path / "chat.md").write_text(_DIALOGO, encoding="utf-8")
        (tmp_path / "doc.md").write_text(_DOCUMENTO, encoding="utf-8")
        out = tmp_path / "out.jsonl"
        with pytest.raises(SystemExit):
            _run_cli(["digestor", "--mode", "auto",
                      "--data", str(tmp_path), "--output", str(out)])


# ---------------------------------------------------------------------------
# F4 — Modo VLM con manifiesto externo (labels + prompt por ejemplo + splits)
# ---------------------------------------------------------------------------

_VLM_RECS = [
    {"id": "m1", "image": "m1.png", "label": 1, "text": "hola", "split": "train"},
    {"id": "m2", "image": "m2.png", "label": 0, "text": "adios", "split": "val"},
    {"id": "m3", "image": "missing3.png", "label": 1, "text": "x", "split": "train"},
]


def _mk_vlm(tmp_path, records, name="manifest.jsonl"):
    """Crea las imágenes referenciadas (solo existencia) + el manifiesto jsonl."""
    for r in records:
        img = r.get("image", "")
        if img and not img.startswith("missing"):
            (tmp_path / img).write_bytes(b"\x89PNG\r\n")
    m = tmp_path / name
    m.write_text("\n".join(_json.dumps(r, ensure_ascii=False) for r in records),
                 encoding="utf-8")
    return m


class TestManifestReader:
    def test_jsonl(self, tmp_path):
        m = _mk_vlm(tmp_path, _VLM_RECS)
        recs = _dig._read_manifest_records(m)
        assert len(recs) == 3 and recs[0]["id"] == "m1"

    def test_json_dict_inyecta_id(self, tmp_path):
        # .json como mapa id→registro (estilo EXIST): la clave se vuelve id
        data = {"a": {"image": "a.png", "label": 1}, "b": {"image": "b.png", "label": 0}}
        m = tmp_path / "man.json"
        m.write_text(_json.dumps(data), encoding="utf-8")
        recs = _dig._read_manifest_records(m)
        assert {r["id"] for r in recs} == {"a", "b"}

    def test_csv(self, tmp_path):
        m = tmp_path / "man.csv"
        m.write_text("id,image,label\nm1,m1.png,1\n", encoding="utf-8")
        recs = _dig._read_manifest_records(m)
        assert recs[0]["label"] == "1"        # csv → strings

    def test_extension_no_soportada_falla(self, tmp_path):
        m = tmp_path / "man.txt"
        m.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            _dig._read_manifest_records(m)


class TestVLMManifest:
    def test_chatml_label_map_prompt_e_higiene(self, tmp_path):
        m = _mk_vlm(tmp_path, _VLM_RECS)
        d = DataDigestor(mode="vlm", label_map={1: "YES", 0: "NO"}, auto_enrich=False)
        d.from_vlm_manifest(m, prompt_template="T:{text}")
        ex = d.get_examples()
        assert len(ex) == 2                    # m3 (imagen ausente) descartado
        user = ex[0]["messages"][0]["content"]
        assert user[0]["type"] == "image" and "m1.png" in user[0]["image"]
        assert user[1]["text"] == "T:hola"     # (b) prompt por ejemplo
        assert ex[0]["messages"][1]["content"] == "YES"   # (a) label_map aplicado

    def test_prompt_field_override(self, tmp_path):
        recs = [{"id": "m1", "image": "m1.png", "label": 1, "prompt": "CUSTOM"}]
        m = _mk_vlm(tmp_path, recs)
        d = DataDigestor(mode="vlm", auto_enrich=False)
        d.from_vlm_manifest(m, prompt_template="ignorada:{text}")
        assert d.get_examples()[0]["messages"][0]["content"][1]["text"] == "CUSTOM"

    def test_prompt_fijo_por_defecto(self, tmp_path):
        recs = [{"id": "m1", "image": "m1.png", "label": 1}]
        m = _mk_vlm(tmp_path, recs)
        d = DataDigestor(mode="vlm", auto_enrich=False)
        d.from_vlm_manifest(m, question="¿Qué muestra?")
        assert d.get_examples()[0]["messages"][0]["content"][1]["text"] == "¿Qué muestra?"

    def test_split_por_campo(self, tmp_path):
        m = _mk_vlm(tmp_path, _VLM_RECS)
        d = DataDigestor(mode="vlm", auto_enrich=False)
        d.from_vlm_manifest(m, split="train")   # train = m1, m3; m3 sin imagen
        assert len(d.get_examples()) == 1

    def test_split_por_fichero(self, tmp_path):
        m = _mk_vlm(tmp_path, _VLM_RECS)
        splits = tmp_path / "splits.json"
        splits.write_text(_json.dumps({"train": ["m1"], "val": ["m2"]}), encoding="utf-8")
        d = DataDigestor(mode="vlm", auto_enrich=False)
        d.from_vlm_manifest(m, split="val", splits_file=splits)
        ex = d.get_examples()
        assert len(ex) == 1 and ex[0]["messages"][1]["content"] == "0"  # sin label_map

    def test_split_inexistente_en_fichero_falla(self, tmp_path):
        m = _mk_vlm(tmp_path, _VLM_RECS)
        splits = tmp_path / "splits.json"
        splits.write_text(_json.dumps({"train": ["m1"]}), encoding="utf-8")
        d = DataDigestor(mode="vlm", auto_enrich=False)
        with pytest.raises(ValueError):
            d.from_vlm_manifest(m, split="holdout", splits_file=splits)

    def test_output_meta(self, tmp_path):
        m = _mk_vlm(tmp_path, _VLM_RECS)
        d = DataDigestor(mode="vlm", label_map={1: "YES", 0: "NO"}, auto_enrich=False)
        d.from_vlm_manifest(m, output="meta", prompt_template="T:{text}")
        ex = d.get_examples()
        assert len(ex) == 2
        assert set(ex[0]) == {"id", "image", "prompt", "label"}
        assert ex[0]["label"] == "YES" and ex[0]["prompt"] == "T:hola"

    def test_require_images_false_mantiene(self, tmp_path):
        m = _mk_vlm(tmp_path, _VLM_RECS)
        d = DataDigestor(mode="vlm", auto_enrich=False)
        d.from_vlm_manifest(m, require_images=False)
        assert len(d.get_examples()) == 3      # m3 se mantiene (ruta sin resolver)

    def test_output_invalido_falla(self, tmp_path):
        m = _mk_vlm(tmp_path, _VLM_RECS)
        d = DataDigestor(mode="vlm", auto_enrich=False)
        with pytest.raises(ValueError):
            d.from_vlm_manifest(m, output="inventado")

    def test_plantilla_llave_desbalanceada_avisa(self, tmp_path):
        # una llave sin cerrar en la plantilla degrada con aviso claro,
        # no con un error críptico de format a media ejecución
        m = _mk_vlm(tmp_path, _VLM_RECS)
        d = DataDigestor(mode="vlm", auto_enrich=False)
        with pytest.raises(ValueError) as exc:
            d.from_vlm_manifest(m, prompt_template="coste 100{ euros")
        assert "prompt_template" in str(exc.value)

    def test_llaves_escapadas_literales(self, tmp_path):
        # {{ y }} → llaves literales, sin tratarse como campo
        m = _mk_vlm(tmp_path, [_VLM_RECS[0]])
        d = DataDigestor(mode="vlm", auto_enrich=False)
        d.from_vlm_manifest(m, prompt_template="usa {{json}} y {text}")
        assert d.get_examples()[0]["messages"][0]["content"][1]["text"] == "usa {json} y hola"


class TestCLIVLM:
    def _mk(self, tmp_path):
        recs = [
            {"id": "m1", "image": "m1.png", "label": 1, "text": "a", "split": "train"},
            {"id": "m2", "image": "m2.png", "label": 0, "text": "b", "split": "holdout"},
        ]
        return _mk_vlm(tmp_path, recs)

    def test_cli_vlm_chatml_train(self, tmp_path):
        m = self._mk(tmp_path)
        out = tmp_path / "train.jsonl"
        # el \n literal de la plantilla debe volverse un salto real en el CLI
        _run_cli(["digestor", "--mode", "vlm", "--manifest", str(m),
                  "--label-col", "label", "--label-map", "1:YES,0:NO",
                  "--prompt-template", "T:{text}\\nQ", "--split", "train",
                  "--output", str(out)])
        rows = _rows(out)
        assert len(rows) == 1
        assert rows[0]["messages"][1]["content"] == "YES"
        assert rows[0]["messages"][0]["content"][1]["text"] == "T:a\nQ"

    def test_cli_vlm_sin_manifest_con_fichero_avisa(self, tmp_path):
        # --mode vlm sin --manifest y con --data no-carpeta: antes despachaba
        # por tipo de fichero (hacía classify en silencio); ahora avisa.
        csv = tmp_path / "datos.csv"
        csv.write_text("text,label\nx,1\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"
        with pytest.raises(SystemExit):
            _run_cli(["digestor", "--mode", "vlm", "--task", "¿Qué ves?",
                      "--data", str(csv), "--output", str(out)])

    def test_cli_vlm_sin_manifest_carpeta_de_imagenes(self, tmp_path):
        # retrocompat: etiquetas por subcarpeta, --task es la pregunta
        (tmp_path / "GATO").mkdir()
        (tmp_path / "PERRO").mkdir()
        (tmp_path / "GATO" / "a.png").write_bytes(b"\x89PNG\r\n")
        (tmp_path / "PERRO" / "b.png").write_bytes(b"\x89PNG\r\n")
        out = tmp_path / "vlm.jsonl"
        _run_cli(["digestor", "--mode", "vlm", "--task", "¿Qué animal es?",
                  "--data", str(tmp_path), "--output", str(out)])
        rows = _rows(out)
        assert len(rows) == 2
        etiquetas = {r["messages"][1]["content"] for r in rows}
        assert etiquetas == {"GATO", "PERRO"}

    def test_cli_vlm_meta_holdout(self, tmp_path):
        m = self._mk(tmp_path)
        out = tmp_path / "holdout.meta.jsonl"
        _run_cli(["digestor", "--mode", "vlm", "--manifest", str(m),
                  "--label-col", "label", "--label-map", "1:YES,0:NO",
                  "--vlm-output", "meta", "--split", "holdout",
                  "--output", str(out)])
        rows = _rows(out)
        assert len(rows) == 1
        assert set(rows[0]) == {"id", "image", "prompt", "label"}
        assert rows[0]["label"] == "NO" and rows[0]["id"] == "m2"
