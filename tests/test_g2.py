"""
tests/test_g2.py
================
Tests para G2: Export universal (LLaMA-Factory, Unsloth, Axolotl).

Verifica los tres métodos de exportación de DataDigestor y el método
auxiliar load_jsonl(), sin necesidad de GPU ni dependencias externas.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motor.digestor import DataDigestor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _digestor_n(n: int) -> DataDigestor:
    """DataDigestor con n ejemplos ChatML de prueba."""
    d = DataDigestor(task="convert")
    for i in range(n):
        d._examples.append({
            "messages": [
                {"role": "system",    "content": f"Eres un asistente. Ejemplo {i}."},
                {"role": "user",      "content": f"Pregunta número {i}"},
                {"role": "assistant", "content": f"Respuesta número {i}"},
            ]
        })
    return d


def _jsonl_file(tmp_path: Path, n: int = 3) -> Path:
    """Crea un JSONL de prueba con n ejemplos ChatML y devuelve su ruta."""
    p = tmp_path / "src.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = {
                "messages": [
                    {"role": "user",      "content": f"msg {i}"},
                    {"role": "assistant", "content": f"resp {i}"},
                ]
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return p


# ===========================================================================
# A: load_jsonl
# ===========================================================================

class TestLoadJsonl:
    def test_carga_ejemplos(self, tmp_path):
        src = _jsonl_file(tmp_path, n=5)
        d = DataDigestor(task="convert")
        d.load_jsonl(src)
        assert len(d.get_examples()) == 5

    def test_devuelve_self(self, tmp_path):
        src = _jsonl_file(tmp_path, n=2)
        d = DataDigestor(task="convert")
        result = d.load_jsonl(src)
        assert result is d

    def test_encadenamiento_llamafactory(self, tmp_path):
        src = _jsonl_file(tmp_path, n=4)
        out = tmp_path / "lf_out"
        n = DataDigestor(task="convert").load_jsonl(src).to_llamafactory(out, dataset_name="chained")
        assert n == 4
        assert (out / "chained.json").exists()

    def test_ejemplos_conservan_contenido(self, tmp_path):
        src = _jsonl_file(tmp_path, n=1)
        d = DataDigestor(task="convert")
        d.load_jsonl(src)
        ex = d.get_examples()[0]
        assert ex["messages"][0]["role"] == "user"
        assert ex["messages"][0]["content"] == "msg 0"

    def test_lineas_vacias_ignoradas(self, tmp_path):
        p = tmp_path / "blank.jsonl"
        p.write_text(
            json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n\n\n",
            encoding="utf-8",
        )
        d = DataDigestor(task="convert")
        d.load_jsonl(p)
        assert len(d.get_examples()) == 1


# ===========================================================================
# B: to_llamafactory
# ===========================================================================

class TestToLlamaFactory:
    def test_crea_json_e_info(self, tmp_path):
        n = _digestor_n(3).to_llamafactory(tmp_path, dataset_name="test_ds")
        assert n == 3
        assert (tmp_path / "test_ds.json").exists()
        assert (tmp_path / "dataset_info.json").exists()

    def test_json_es_lista(self, tmp_path):
        _digestor_n(2).to_llamafactory(tmp_path, dataset_name="ds")
        data = json.loads((tmp_path / "ds.json").read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 2

    def test_estructura_sharegpt(self, tmp_path):
        _digestor_n(2).to_llamafactory(tmp_path, dataset_name="ds")
        data = json.loads((tmp_path / "ds.json").read_text(encoding="utf-8"))
        ex = data[0]
        assert "conversations" in ex
        assert all("from" in c and "value" in c for c in ex["conversations"])

    def test_mapeo_roles(self, tmp_path):
        d = DataDigestor(task="convert")
        d._examples.append({
            "messages": [
                {"role": "system",    "content": "sys"},
                {"role": "user",      "content": "usr"},
                {"role": "assistant", "content": "asst"},
            ]
        })
        d.to_llamafactory(tmp_path, dataset_name="r")
        data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        by_role = {c["from"]: c["value"] for c in data[0]["conversations"]}
        assert by_role["system"] == "sys"
        assert by_role["human"] == "usr"
        assert by_role["gpt"] == "asst"

    def test_dataset_info_keys(self, tmp_path):
        _digestor_n(2).to_llamafactory(tmp_path, dataset_name="my_ds")
        info = json.loads((tmp_path / "dataset_info.json").read_text(encoding="utf-8"))
        assert "my_ds" in info
        meta = info["my_ds"]
        assert meta["file_name"] == "my_ds.json"
        assert meta["formatting"] == "sharegpt"

    def test_dataset_info_tags(self, tmp_path):
        _digestor_n(1).to_llamafactory(tmp_path, dataset_name="t")
        info = json.loads((tmp_path / "dataset_info.json").read_text(encoding="utf-8"))
        tags = info["t"]["tags"]
        assert tags["user_tag"] == "human"
        assert tags["assistant_tag"] == "gpt"
        assert tags["role_tag"] == "from"

    def test_vacio_devuelve_cero(self, tmp_path):
        n = DataDigestor(task="convert").to_llamafactory(tmp_path)
        assert n == 0

    def test_nombre_por_defecto(self, tmp_path):
        _digestor_n(1).to_llamafactory(tmp_path)
        assert (tmp_path / "mi_dataset.json").exists()
        assert (tmp_path / "dataset_info.json").exists()


# ===========================================================================
# C: to_unsloth
# ===========================================================================

class TestToUnsloth:
    def test_crea_jsonl(self, tmp_path):
        out = tmp_path / "unsloth.jsonl"
        n = _digestor_n(4).to_unsloth(out)
        assert n == 4
        assert out.exists()

    def test_formato_alpaca(self, tmp_path):
        out = tmp_path / "u.jsonl"
        _digestor_n(2).to_unsloth(out)
        lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2
        ex = lines[0]
        assert "instruction" in ex
        assert "output" in ex

    def test_output_no_vacio(self, tmp_path):
        out = tmp_path / "u.jsonl"
        _digestor_n(3).to_unsloth(out)
        lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert all(ex["output"] for ex in lines)

    def test_todos_los_ejemplos(self, tmp_path):
        out = tmp_path / "u.jsonl"
        n = _digestor_n(10).to_unsloth(out)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert n == len(lines) == 10

    def test_vacio_devuelve_cero(self, tmp_path):
        n = DataDigestor(task="convert").to_unsloth(tmp_path / "empty.jsonl")
        assert n == 0

    def test_crea_directorio_padre(self, tmp_path):
        out = tmp_path / "subdir" / "unsloth.jsonl"
        _digestor_n(2).to_unsloth(out)
        assert out.exists()


# ===========================================================================
# D: to_axolotl
# ===========================================================================

class TestToAxolotl:
    def test_crea_jsonl_y_yaml(self, tmp_path):
        n = _digestor_n(3).to_axolotl(tmp_path, dataset_name="ax_test")
        assert n == 3
        assert (tmp_path / "ax_test.jsonl").exists()
        assert (tmp_path / "axolotl_config.yml").exists()

    def test_formato_sharegpt(self, tmp_path):
        _digestor_n(2).to_axolotl(tmp_path, dataset_name="ax")
        lines = [
            json.loads(l)
            for l in (tmp_path / "ax.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert len(lines) == 2
        ex = lines[0]
        assert "conversations" in ex
        assert any(c["from"] == "human" for c in ex["conversations"])
        assert any(c["from"] == "gpt" for c in ex["conversations"])

    def test_yaml_contiene_sharegpt(self, tmp_path):
        _digestor_n(1).to_axolotl(tmp_path, dataset_name="myds")
        yaml_text = (tmp_path / "axolotl_config.yml").read_text(encoding="utf-8")
        assert "sharegpt" in yaml_text

    def test_yaml_contiene_nombre_dataset(self, tmp_path):
        _digestor_n(1).to_axolotl(tmp_path, dataset_name="mi_ds_axolotl")
        yaml_text = (tmp_path / "axolotl_config.yml").read_text(encoding="utf-8")
        assert "mi_ds_axolotl" in yaml_text

    def test_todos_los_ejemplos(self, tmp_path):
        n = _digestor_n(7).to_axolotl(tmp_path, dataset_name="ax")
        lines = [
            l for l in (tmp_path / "ax.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert n == len(lines) == 7

    def test_vacio_devuelve_cero(self, tmp_path):
        n = DataDigestor(task="convert").to_axolotl(tmp_path)
        assert n == 0

    def test_nombre_por_defecto(self, tmp_path):
        _digestor_n(1).to_axolotl(tmp_path)
        assert (tmp_path / "mi_dataset.jsonl").exists()


# ===========================================================================
# E: Conversión completa — round-trip load_jsonl → export
# ===========================================================================

class TestRoundTrip:
    def test_load_then_llamafactory(self, tmp_path):
        src = _jsonl_file(tmp_path, n=5)
        out = tmp_path / "lf"
        n = DataDigestor(task="convert").load_jsonl(src).to_llamafactory(out, dataset_name="rt")
        assert n == 5
        data = json.loads((out / "rt.json").read_text(encoding="utf-8"))
        assert len(data) == 5

    def test_load_then_unsloth(self, tmp_path):
        src = _jsonl_file(tmp_path, n=4)
        out = tmp_path / "unsloth.jsonl"
        n = DataDigestor(task="convert").load_jsonl(src).to_unsloth(out)
        assert n == 4

    def test_load_then_axolotl(self, tmp_path):
        src = _jsonl_file(tmp_path, n=6)
        out = tmp_path / "ax_out"
        n = DataDigestor(task="convert").load_jsonl(src).to_axolotl(out, dataset_name="rt")
        assert n == 6
        assert (out / "rt.jsonl").exists()
        assert (out / "axolotl_config.yml").exists()

    def test_dataset_domestic_v2_llamafactory(self, tmp_path):
        """Si existe el dataset real, verifica que se convierte sin errores."""
        src = Path("datasets/dataset_domestic_v2.jsonl")
        if not src.exists():
            pytest.skip("dataset_domestic_v2.jsonl no disponible")
        out = tmp_path / "lf_domestic"
        n = DataDigestor(task="convert").load_jsonl(src).to_llamafactory(out, dataset_name="domestic")
        assert n > 0
        data = json.loads((out / "domestic.json").read_text(encoding="utf-8"))
        assert len(data) == n
