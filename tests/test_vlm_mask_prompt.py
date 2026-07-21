"""
test_vlm_mask_prompt.py
=======================
Completion-only loss en el collator VLM (`_VLMDataCollator(mask_prompt=True)`).

Sin enmascarado, la pérdida cubre prompt + tokens de imagen + respuesta. En
tareas de respuesta corta (una etiqueta YES/NO frente a un prompt de cientos de
tokens) la señal útil queda diluida ~1:300. Estos tests fijan el contrato:

  - por defecto NO se enmascara (comportamiento histórico intacto),
  - con mask_prompt=True solo la respuesta contribuye a la pérdida,
  - el conteo del prompt se hace CON la imagen (los VLM la expanden a N tokens),
  - si la truncación se come la respuesta, se degrada sin romper la pérdida.

Usa un processor falso y determinista (1 palabra = 1 token) para poder afirmar
posiciones exactas sin descargar un VLM.
"""

import torch

from motor.trainer_vlm import _VLMDataCollator

PAD_ID = 0
IMG_TOKENS = 4  # el processor falso expande cada imagen a 4 tokens


class _FakeTokenizer:
    pad_token_id = PAD_ID
    padding_side = "right"
    eos_token = "</s>"


class _FakeProcessor:
    """
    Tokeniza por palabras. Cada imagen aporta IMG_TOKENS tokens extra, igual
    que un VLM real expande el placeholder según la resolución.
    """

    def __init__(self, padding_side="right"):
        self.tokenizer = _FakeTokenizer()
        self.tokenizer.padding_side = padding_side
        self._vocab = {"<pad>": PAD_ID}

    def _id(self, word):
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab) + 1
        return self._vocab[word]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            parts.append(f"{m['role']}: {content}")
        text = " ".join(parts)
        if add_generation_prompt:
            text += " assistant:"
        return text

    def __call__(self, text=None, images=None, padding=False, truncation=False,
                 max_length=None, return_tensors=None):
        seqs = []
        for i, t in enumerate(text):
            ids = [self._id(w) for w in t.split()]
            n_img = 1 if images and i < len(images) and images[i] is not None else 0
            ids = [self._id("<img>")] * (IMG_TOKENS * n_img) + ids
            if truncation and max_length:
                ids = ids[:max_length]
            seqs.append(ids)

        width = max(len(s) for s in seqs)
        rows = []
        for s in seqs:
            pad = [PAD_ID] * (width - len(s))
            rows.append(pad + s if self.tokenizer.padding_side == "left" else s + pad)
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}


def _example(answer="YES", prompt="es esto sexista"):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": "/no/existe.jpg"},  # → placeholder blanco
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": answer},
        ]
    }


# --- por defecto: nada cambia -------------------------------------------------

def test_por_defecto_no_enmascara_el_prompt():
    col = _VLMDataCollator(_FakeProcessor(), max_seq_length=64)
    batch = col([_example()])

    labels, ids = batch["labels"], batch["input_ids"]
    # solo el padding va a -100; el resto conserva el token
    assert torch.equal(labels[ids != PAD_ID], ids[ids != PAD_ID])
    assert (labels == -100).sum() == (ids == PAD_ID).sum()


# --- con mask_prompt: solo la respuesta puntúa --------------------------------

def test_mask_prompt_deja_solo_la_respuesta():
    proc = _FakeProcessor()
    col = _VLMDataCollator(proc, max_seq_length=64, mask_prompt=True)
    ex = _example(answer="YES")
    batch = col([ex])

    labels = batch["labels"][0]
    ids = batch["input_ids"][0]

    vivos = (labels != -100).nonzero().flatten().tolist()
    assert len(vivos) == 1, f"solo el token de respuesta debe puntuar, vivos={vivos}"
    assert ids[vivos[0]].item() == proc._vocab["YES"]


def test_mask_prompt_cuenta_los_tokens_de_imagen():
    """
    El corte debe caer DESPUÉS de los tokens de imagen. Si se contara el prompt
    sin la imagen, el corte quedaría IMG_TOKENS posiciones antes y dejaría
    tokens de prompt sin enmascarar.
    """
    proc = _FakeProcessor()
    col = _VLMDataCollator(proc, max_seq_length=64, mask_prompt=True)
    batch = col([_example()])

    labels = batch["labels"][0]
    corte = (labels != -100).nonzero().flatten()[0].item()
    assert corte >= IMG_TOKENS, f"el corte {corte} ignora los tokens de imagen"


def test_mask_prompt_respeta_respuestas_multitoken():
    proc = _FakeProcessor()
    col = _VLMDataCollator(proc, max_seq_length=64, mask_prompt=True)
    batch = col([_example(answer="claramente si")])

    labels = batch["labels"][0]
    assert (labels != -100).sum().item() == 2


# --- degradación: nunca dejar la pérdida en NaN -------------------------------

def test_si_la_truncacion_se_come_la_respuesta_no_enmascara():
    """
    Con max_seq_length corto la respuesta se pierde en la truncación. Enmascarar
    dejaría TODAS las labels a -100 → loss NaN. Debe degradar sin enmascarar.
    """
    proc = _FakeProcessor()
    col = _VLMDataCollator(proc, max_seq_length=IMG_TOKENS + 2, mask_prompt=True)
    batch = col([_example(prompt="uno dos tres cuatro cinco seis siete ocho")])

    labels = batch["labels"][0]
    assert not bool((labels == -100).all()), "no debe quedar el ejemplo entero a -100"


# --- batch con padding --------------------------------------------------------

def test_batch_mixto_con_padding_derecha():
    proc = _FakeProcessor(padding_side="right")
    col = _VLMDataCollator(proc, max_seq_length=64, mask_prompt=True)
    batch = col([
        _example(prompt="corto"),
        _example(prompt="un prompt bastante mas largo que el otro"),
    ])

    for fila in batch["labels"]:
        assert (fila != -100).sum().item() == 1


class _FakeProcessorEosIgualPad(_FakeProcessor):
    """
    Modelos que reutilizan el token de pad como eos. La secuencia acaba en un
    token idéntico al de relleno, así que contar "pads totales" para deducir la
    longitud real desplaza el corte y se come el primer token de la respuesta.
    Hay que contar solo los pads de CABECERA.
    """

    def __call__(self, **kw):
        out = super().__call__(**kw)
        # El eos solo cierra la secuencia de ENTRENAMIENTO. La codificación del
        # prompt (add_generation_prompt, sin padding) no lo lleva: si no, se
        # inflaría n_prompt y el corte se desplazaría por culpa del test.
        if not kw.get("padding"):
            return out
        ids = out["input_ids"]
        eos = torch.full((ids.shape[0], 1), PAD_ID, dtype=torch.long)
        return {"input_ids": torch.cat([ids, eos], dim=1)}


def test_eos_igual_a_pad_no_desplaza_el_corte():
    proc = _FakeProcessorEosIgualPad(padding_side="left")
    col = _VLMDataCollator(proc, max_seq_length=64, mask_prompt=True)
    batch = col([
        _example(answer="claramente si", prompt="corto"),
        _example(answer="claramente si", prompt="un prompt bastante mas largo"),
    ])

    ids = batch["input_ids"]
    for i, fila in enumerate(batch["labels"]):
        vivos = (fila != -100).nonzero().flatten().tolist()
        assert len(vivos) == 2, (
            f"fila {i}: se esperaban los 2 tokens de respuesta, hay {len(vivos)} "
            f"(¿el corte se comió el primero por contar el eos como padding?)"
        )
        assert ids[i, vivos[0]].item() == proc._vocab["claramente"]


def test_padding_izquierda_no_desalinea_el_corte():
    proc = _FakeProcessor(padding_side="left")
    col = _VLMDataCollator(proc, max_seq_length=64, mask_prompt=True)
    batch = col([
        _example(prompt="corto"),
        _example(prompt="un prompt bastante mas largo que el otro"),
    ])

    ids = batch["input_ids"]
    for i, fila in enumerate(batch["labels"]):
        vivos = (fila != -100).nonzero().flatten().tolist()
        assert len(vivos) == 1, f"fila {i}: vivos={vivos}"
        assert ids[i, vivos[0]].item() == proc._vocab["YES"]
