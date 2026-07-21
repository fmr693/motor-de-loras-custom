"""
test_vlm_keep_best.py
=====================
El VLMTrainer debe guardar la MEJOR época, no la última.

Antes: `eval_strategy="epoch"` + `save_strategy="no"` sin `load_best_model_at_end`
→ el Trainer medía el sobreajuste cada época y lo ignoraba. Medido en el hito
EXIST-VLM: se guardó un adapter de eval_loss 0.2652 habiendo pasado por 0.1658
(trayectoria 0.1658 → 0.1850 → 0.2652), y ese adapter rindió peor en el holdout.

Estos tests fijan el contrato sin entrenar nada: la decisión vive en el helper
`_args_seleccion_checkpoint`, testeable en aislamiento.
"""

import inspect

from motor.trainer_vlm import VLMTrainer, _args_seleccion_checkpoint


class _DatasetFalso:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


# --- caso normal: hay con qué comparar -----------------------------------------

def test_con_eval_guarda_la_mejor_epoca():
    args = _args_seleccion_checkpoint(keep_best=True, eval_ds=_DatasetFalso(138))

    assert args["load_best_model_at_end"] is True
    assert args["save_strategy"] == "epoch", "sin guardar por época no hay dónde volver"
    assert args["metric_for_best_model"] == "eval_loss"
    assert args["greater_is_better"] is False, "en pérdida, menos es mejor"


def test_save_strategy_acompana_a_eval_strategy():
    """
    load_best_model_at_end exige que save_strategy case con eval_strategy
    ("epoch"); si no, transformers lanza al construir TrainingArguments.
    """
    args = _args_seleccion_checkpoint(keep_best=True, eval_ds=_DatasetFalso(10))
    assert args["save_strategy"] == "epoch"


# --- degradaciones: nunca fingir que se puede elegir ----------------------------

def test_sin_eval_dataset_guarda_la_ultima_y_avisa(capsys):
    args = _args_seleccion_checkpoint(keep_best=True, eval_ds=_DatasetFalso(0))

    assert args["save_strategy"] == "no"
    assert "load_best_model_at_end" not in args
    assert "AVISO" in capsys.readouterr().out


def test_eval_none_guarda_la_ultima_y_avisa(capsys):
    args = _args_seleccion_checkpoint(keep_best=True, eval_ds=None)

    assert args["save_strategy"] == "no"
    assert "load_best_model_at_end" not in args
    assert "AVISO" in capsys.readouterr().out


def test_desactivado_no_guarda_checkpoints_ni_avisa(capsys):
    """keep_best=False es una elección explícita: no hay nada que advertir."""
    args = _args_seleccion_checkpoint(keep_best=False, eval_ds=_DatasetFalso(138))

    assert args["save_strategy"] == "no"
    assert "load_best_model_at_end" not in args
    assert "AVISO" not in capsys.readouterr().out


def test_desactivado_sin_eval_tampoco_avisa(capsys):
    args = _args_seleccion_checkpoint(keep_best=False, eval_ds=None)

    assert args["save_strategy"] == "no"
    assert "AVISO" not in capsys.readouterr().out


# --- resolución fina: elegir el mejor PUNTO, no la mejor época -----------------

def test_eval_por_pasos_ajusta_ambas_estrategias():
    """
    Con eval_every_steps, evaluación Y guardado deben pasar a "steps" con el
    mismo intervalo: si solo cambiase una, transformers rechaza
    load_best_model_at_end por estrategias incompatibles.
    """
    args = _args_seleccion_checkpoint(True, _DatasetFalso(138), eval_every_steps=40)

    assert args["eval_strategy"] == "steps"
    assert args["save_strategy"] == "steps"
    assert args["eval_steps"] == args["save_steps"] == 40
    assert args["load_best_model_at_end"] is True


def test_sin_eval_every_steps_sigue_siendo_por_epoca():
    """El comportamiento por defecto no cambia: por época."""
    args = _args_seleccion_checkpoint(True, _DatasetFalso(138))

    assert args["eval_strategy"] == "epoch"
    assert args["save_strategy"] == "epoch"
    assert "eval_steps" not in args


def test_eval_por_pasos_se_respeta_aunque_no_se_guarde():
    """
    keep_best=False no debe anular el ritmo de evaluación pedido: son cosas
    distintas (cada cuánto se MIDE vs con qué punto se QUEDA).
    """
    args = _args_seleccion_checkpoint(False, _DatasetFalso(138), eval_every_steps=25)

    assert args["eval_strategy"] == "steps"
    assert args["eval_steps"] == 25
    assert args["save_strategy"] == "no"


def test_eval_every_steps_cero_o_negativo_cae_a_epoca():
    for valor in (0, -10):
        args = _args_seleccion_checkpoint(True, _DatasetFalso(138), eval_every_steps=valor)
        assert args["eval_strategy"] == "epoch", f"eval_every_steps={valor}"


# --- contra transformers REAL, no contra un doble ------------------------------

def test_transformers_acepta_la_combinacion(tmp_path):
    """
    `load_best_model_at_end=True` impone condiciones cruzadas (save_strategy debe
    casar con eval_strategy, la métrica debe existir). Un test con dobles diría
    que sí aunque transformers lo rechazara al construir los argumentos, así que
    esto se comprueba contra la librería de verdad.
    """
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir       = str(tmp_path),
        num_train_epochs = 3,
        report_to        = "none",
        **_args_seleccion_checkpoint(keep_best=True, eval_ds=_DatasetFalso(138)),
    )

    assert args.load_best_model_at_end is True
    assert args.greater_is_better is False
    assert str(args.save_strategy).lower().endswith("epoch")


def test_transformers_acepta_la_combinacion_por_pasos(tmp_path):
    """Mismo contrato con la política por pasos, que es la que cruza más flags."""
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir       = str(tmp_path),
        num_train_epochs = 3,
        report_to        = "none",
        **_args_seleccion_checkpoint(True, _DatasetFalso(138), eval_every_steps=40),
    )

    assert args.load_best_model_at_end is True
    assert args.eval_steps == 40
    assert args.save_steps == 40
    assert str(args.save_strategy).lower().endswith("steps")


# --- el defecto importa: quedarse con la mejor debe ser lo que pasa sin pedirlo --

def test_fit_activa_keep_best_por_defecto():
    assert inspect.signature(VLMTrainer.fit).parameters["keep_best"].default is True
