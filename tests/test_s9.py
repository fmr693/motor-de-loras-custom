"""
tests/test_s9.py
================
Sprint 9 — User Personalization Layer.

Tests para:
  A. DataDigestor.from_user_profile()
  B. ContinualLearner.stack_adapters()  (con torch/peft mockeados)

Ninguno requiere GPU ni descarga de modelos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from motor.digestor import DataDigestor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROFILE_ES = {
    "name":         "Marta",
    "work_dirs":    ["~/Desktop/Proyecto_V3", "~/Documentos/Trabajo"],
    "notes_folder": "~/Notas",
    "contacts":     {"jefe": "jefe@empresa.com", "equipo": "equipo@empresa.com"},
    "language":     "es",
    "common_tasks": ["organizar las descargas", "revisar el presupuesto"],
}

PROFILE_EN = {
    "name":         "Alice",
    "work_dirs":    ["~/Documents/Work"],
    "notes_folder": "~/Notes",
    "contacts":     {"boss": "boss@company.com"},
    "language":     "en",
    "common_tasks": ["organize downloads"],
}


def _write_profile(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "user_profile.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ===========================================================================
# A. DataDigestor.from_user_profile()
# ===========================================================================

class TestFromUserProfile:

    # -- Carga básica --------------------------------------------------------

    def test_genera_ejemplos(self, tmp_path):
        """from_user_profile() debe generar ejemplos (n_per_tool * 5 herramientas + common_tasks)."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=3)
        # 5 herramientas × 3 + 2 common_tasks
        assert len(d._examples) == 5 * 3 + 2

    def test_genera_ejemplos_en(self, tmp_path):
        """Versión inglés genera las mismas cantidades."""
        p = _write_profile(tmp_path, PROFILE_EN)
        d = DataDigestor(task="personal agent")
        d.from_user_profile(str(p), n_per_tool=4)
        # 5 herramientas × 4 + 1 common_task
        assert len(d._examples) == 5 * 4 + 1

    def test_formato_chatml(self, tmp_path):
        """Cada ejemplo tiene la estructura messages con roles correctos."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=2)
        for ex in d._examples:
            assert "messages" in ex
            roles = [m["role"] for m in ex["messages"]]
            assert "system" in roles
            assert "user" in roles
            assert "assistant" in roles

    def test_react_format(self, tmp_path):
        """En formato react, el assistant contiene 'Thought:' y 'Final Answer:'."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=2, format="react")
        for ex in d._examples:
            assistant = next(
                m["content"] for m in ex["messages"] if m["role"] == "assistant"
            )
            assert "Thought:" in assistant
            assert "Final Answer:" in assistant

    def test_function_call_format(self, tmp_path):
        """En formato function_call, el assistant es JSON con 'tool' y 'args'."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=2, format="function_call")
        for ex in d._examples:
            assistant = next(
                m["content"] for m in ex["messages"] if m["role"] == "assistant"
            )
            parsed = json.loads(assistant)
            assert "tool" in parsed
            assert "args" in parsed

    # -- Personalización real ------------------------------------------------

    def test_rutas_personalizadas_en_ejemplos(self, tmp_path):
        """El trabajo del usuario (work_dirs) aparece en al menos un ejemplo."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=5)
        all_text = " ".join(
            " ".join(m["content"] for m in ex["messages"])
            for ex in d._examples
        )
        # Al menos una ruta del perfil debe aparecer en el texto generado
        found = any(wd in all_text for wd in PROFILE_ES["work_dirs"])
        assert found, "Ninguna ruta de work_dirs aparece en los ejemplos"

    def test_notas_folder_en_args(self, tmp_path):
        """Los ejemplos note_save usan notes_folder del perfil."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=3, format="react")
        # El nombre de la carpeta de notas debe aparecer en al menos un ejemplo
        notes_name = Path(PROFILE_ES["notes_folder"]).name  # "Notas"
        all_text = " ".join(
            " ".join(m["content"] for m in ex["messages"])
            for ex in d._examples
        )
        assert notes_name in all_text

    def test_contactos_en_email_ejemplos(self, tmp_path):
        """Los nombres de contacto del perfil aparecen en ejemplos de email_filter."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=5)
        all_text = " ".join(
            " ".join(m["content"] for m in ex["messages"])
            for ex in d._examples
        )
        found = any(c in all_text for c in PROFILE_ES["contacts"])
        assert found, "Ningún contacto del perfil aparece en los ejemplos"

    def test_common_tasks_como_ejemplos(self, tmp_path):
        """Cada common_task genera exactamente un ejemplo extra."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=1)
        base = 5 * 1
        extras = len(PROFILE_ES["common_tasks"])
        assert len(d._examples) == base + extras

    def test_chaining_to_jsonl(self, tmp_path):
        """Se puede encadenar con .to_jsonl() sin error."""
        p = _write_profile(tmp_path, PROFILE_ES)
        out = tmp_path / "personal.jsonl"
        d = DataDigestor(task="agente personal")
        n = d.from_user_profile(str(p), n_per_tool=2).to_jsonl(str(out))
        assert out.exists()
        assert n > 0

    def test_perfil_no_encontrado_lanza_error(self, tmp_path):
        """Si el archivo no existe, lanza FileNotFoundError."""
        d = DataDigestor(task="agente personal")
        with pytest.raises(FileNotFoundError):
            d.from_user_profile(str(tmp_path / "no_existe.json"))

    def test_sin_common_tasks(self, tmp_path):
        """Perfil sin common_tasks genera solo n_per_tool * 5 ejemplos."""
        profile = {**PROFILE_ES, "common_tasks": []}
        p = _write_profile(tmp_path, profile)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=3)
        assert len(d._examples) == 5 * 3

    def test_sistema_menciona_nombre_usuario(self, tmp_path):
        """El system prompt menciona el nombre del usuario del perfil."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=2)
        system_contents = [
            m["content"] for ex in d._examples
            for m in ex["messages"] if m["role"] == "system"
        ]
        assert all("Marta" in c for c in system_contents)

    def test_seed_reproducible(self, tmp_path):
        """Mismo seed → mismos ejemplos."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d1 = DataDigestor(task="agente personal")
        d1.from_user_profile(str(p), n_per_tool=3, seed=99)
        d2 = DataDigestor(task="agente personal")
        d2.from_user_profile(str(p), n_per_tool=3, seed=99)
        texts1 = [ex["messages"][-1]["content"] for ex in d1._examples]
        texts2 = [ex["messages"][-1]["content"] for ex in d2._examples]
        assert texts1 == texts2

    def test_seed_diferente_genera_variacion(self, tmp_path):
        """Seeds distintos → al menos un ejemplo diferente."""
        p = _write_profile(tmp_path, PROFILE_ES)
        d1 = DataDigestor(task="agente personal")
        d1.from_user_profile(str(p), n_per_tool=5, seed=1)
        d2 = DataDigestor(task="agente personal")
        d2.from_user_profile(str(p), n_per_tool=5, seed=999)
        t1 = set(ex["messages"][1]["content"] for ex in d1._examples)
        t2 = set(ex["messages"][1]["content"] for ex in d2._examples)
        # Con seeds distintos debe haber alguna diferencia
        assert t1 != t2

    # -- Integración con validate() -----------------------------------------

    def test_validate_post_generacion(self, tmp_path):
        """El semáforo funciona tras from_user_profile()."""
        profile = {**PROFILE_ES, "common_tasks": [f"tarea {i}" for i in range(40)]}
        p = _write_profile(tmp_path, profile)
        d = DataDigestor(task="agente personal")
        d.from_user_profile(str(p), n_per_tool=10)
        report = d.validate(verbose=False)
        assert "semaforo" in report
        assert report["total"] > 0


# ===========================================================================
# B. ContinualLearner.stack_adapters()  — con mocks de torch/peft
# ===========================================================================

class TestStackAdapters:
    """
    Tests de stack_adapters() con torch y peft completamente mockeados.
    No requieren GPU ni descargas.
    """

    def _mock_modules(self):
        """Retorna un dict de mocks para patchear torch y peft."""
        # --- torch mock ---
        mock_torch = MagicMock()
        mock_torch.float16 = "float16"

        # --- modelo mock ---
        mock_model = MagicMock()
        mock_model.save_pretrained = MagicMock()

        # --- PeftModel mock ---
        mock_peft_model = MagicMock()
        mock_peft_model.from_pretrained = MagicMock(return_value=mock_model)
        mock_model.load_adapter = MagicMock()
        mock_model.add_weighted_adapter = MagicMock()
        mock_model.set_adapter = MagicMock()

        # --- AutoModelForCausalLM mock ---
        mock_auto = MagicMock()
        mock_auto.from_pretrained = MagicMock(return_value=mock_model)

        return {
            "torch":                          mock_torch,
            "transformers":                   MagicMock(AutoModelForCausalLM=mock_auto),
            "transformers.AutoModelForCausalLM": mock_auto,
            "peft":                           MagicMock(PeftModel=mock_peft_model),
            "peft.PeftModel":                 mock_peft_model,
        }

    def _make_adapter_dir(self, tmp_path: Path, name: str) -> Path:
        """Crea un directorio falso de adapter con adapter_config.json y meta.json."""
        d = tmp_path / name
        d.mkdir()
        (d / "adapter_config.json").write_text(
            json.dumps({"r": 16, "lora_alpha": 32}), encoding="utf-8"
        )
        (d / "meta.json").write_text(
            json.dumps({"base_model": "Qwen/Qwen2.5-3B-Instruct", "eval_loss": 0.1}),
            encoding="utf-8",
        )
        return d

    def test_stack_crea_output_dir(self, tmp_path):
        """stack_adapters() debe crear el directorio de salida."""
        from motor.continual import ContinualLearner

        base     = self._make_adapter_dir(tmp_path, "base")
        personal = self._make_adapter_dir(tmp_path, "personal")
        out      = tmp_path / "stacked"

        mocks = self._mock_modules()
        with patch.dict("sys.modules", mocks):
            cl = ContinualLearner(model_id="Qwen/Qwen2.5-3B-Instruct")
            cl.stack_adapters(str(base), str(personal), str(out))

        assert out.exists()

    def test_stack_genera_meta_json(self, tmp_path):
        """stack_adapters() debe guardar un meta.json con la info de la combinación."""
        from motor.continual import ContinualLearner

        base     = self._make_adapter_dir(tmp_path, "base")
        personal = self._make_adapter_dir(tmp_path, "personal")
        out      = tmp_path / "stacked"

        mocks = self._mock_modules()
        with patch.dict("sys.modules", mocks):
            cl = ContinualLearner(model_id="Qwen/Qwen2.5-3B-Instruct")
            cl.stack_adapters(str(base), str(personal), str(out),
                              base_weight=0.8, personal_weight=0.2)

        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["stacked"] is True
        assert meta["base_weight"] == 0.8
        assert meta["personal_weight"] == 0.2
        assert meta["model_id"] == "Qwen/Qwen2.5-3B-Instruct"

    def test_stack_devuelve_ruta(self, tmp_path):
        """stack_adapters() debe devolver la ruta de salida como string."""
        from motor.continual import ContinualLearner

        base     = self._make_adapter_dir(tmp_path, "base")
        personal = self._make_adapter_dir(tmp_path, "personal")
        out      = tmp_path / "stacked"

        mocks = self._mock_modules()
        with patch.dict("sys.modules", mocks):
            cl = ContinualLearner(model_id="Qwen/Qwen2.5-3B-Instruct")
            result = cl.stack_adapters(str(base), str(personal), str(out))

        assert result == str(out)

    def test_stack_copia_tokenizer(self, tmp_path):
        """Si el adapter base tiene carpeta tokenizer/, se copia al output."""
        from motor.continual import ContinualLearner

        base = self._make_adapter_dir(tmp_path, "base")
        tok_dir = base / "tokenizer"
        tok_dir.mkdir()
        (tok_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        personal = self._make_adapter_dir(tmp_path, "personal")
        out      = tmp_path / "stacked"

        mocks = self._mock_modules()
        with patch.dict("sys.modules", mocks):
            cl = ContinualLearner(model_id="Qwen/Qwen2.5-3B-Instruct")
            cl.stack_adapters(str(base), str(personal), str(out))

        assert (out / "tokenizer" / "tokenizer_config.json").exists()

    def test_stack_pesos_por_defecto(self, tmp_path):
        """Los pesos por defecto son 0.7 (base) y 0.3 (personal)."""
        from motor.continual import ContinualLearner

        base     = self._make_adapter_dir(tmp_path, "base")
        personal = self._make_adapter_dir(tmp_path, "personal")
        out      = tmp_path / "stacked"

        mocks = self._mock_modules()
        with patch.dict("sys.modules", mocks):
            cl = ContinualLearner(model_id="Qwen/Qwen2.5-3B-Instruct")
            cl.stack_adapters(str(base), str(personal), str(out))

        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["base_weight"] == 0.7
        assert meta["personal_weight"] == 0.3
