"""
motor.server
============
Servidor REST para la Fábrica de LoRAs.

Expone cualquier adapter LoRA o modelo HuggingFace como API HTTP con FastAPI.
El modelo se carga UNA sola vez al arrancar y queda en memoria para responder
peticiones en milisegundos — base necesaria para el sistema agéntico.

Endpoints
---------
  GET  /health               → estado del servidor, GPU, modelo cargado, VRAM
  GET  /v1/models            → lista de modelos (OpenAI-compatible)
  POST /v1/chat/completions  → chat OpenAI-compatible: SSE streaming, tools,
                               logging de interacciones (id = interaction_id)
  POST /chat                 → inferencia texto (stateless, sin historial)
  POST /chat/session         → inferencia con historial de sesión (stateful)
  DELETE /chat/session       → borrar historial de una sesión
  POST /feedback             → 👍/👎 sobre una interacción (alimenta DPO)
  POST /agent                → agente ReAct con herramientas (requiere API key)

Inicio rápido
-------------
  python fabrica_loras.py serve --model adapters/finance_sentiment_14b/

  # Desde cualquier cliente:
  curl -X POST http://localhost:8000/chat \\
       -H "Content-Type: application/json" \\
       -d '{"message": "AAPL is crashing today"}'

Dependencias
------------
  pip install fastapi uvicorn

Seguridad
---------
  Por defecto solo escucha en localhost (127.0.0.1).
  Para exponer en red local usa --host 0.0.0.0 (solo en redes confiables).
  Opcionalmente protege con --api-key (Bearer token).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Estado global del servidor (cargado una vez al arrancar)
# ---------------------------------------------------------------------------

# Limite de mensajes por sesion para evitar crecimiento infinito en RAM
MAX_SESSION_MESSAGES = 100  # ~50 turnos de conversacion (user + assistant)

# TTL de sesiones inactivas y tope global — un servidor 24/7 acumulaba
# sesiones en RAM sin límite (fuga lenta de memoria)
SESSION_TTL_S = 24 * 3600   # sesiones sin actividad en 24h se descartan
MAX_SESSIONS  = 500         # tope duro: si se supera, se borran las más antiguas

class _ServerState:
    model          = None
    tokenizer      = None
    llama_model    = None   # llama-cpp-python (GGUF mode)
    is_gguf: bool  = False
    # Visión activa: el GGUF se cargó con un mmproj (ver MOTOR_MMPROJ).
    vision_ready: bool = False
    model_path: str = ""
    base_model_id: str = ""
    is_adapter: bool = False
    load_time_s: float = 0.0
    dtype_str: str = ""
    gpu_name: str  = ""
    vram_total_gb: float = 0.0
    api_key: Optional[str] = None
    # Historial de sesiones: session_id → list[dict]
    sessions: Dict[str, List[dict]] = {}
    # Último uso de cada sesión (epoch s) — para purga por TTL
    session_last_seen: Dict[str, float] = {}
    # S10.1 — ruta al archivo de log de interacciones
    # En Docker: /app/logs/ (bind mount compartido con worker)
    # En local:  logs/interaction_log.jsonl relativo al cwd
    interaction_log_path: Optional[str] = "logs/interaction_log.jsonl"


_state = _ServerState()

# ---------------------------------------------------------------------------
# Serialización de la inferencia
# ---------------------------------------------------------------------------
# llama-cpp-python NO es thread-safe: un único contexto compartido entre hilos
# revienta el proceso. FastAPI despacha los endpoints `def` (síncronos) en un
# threadpool, así que DOS peticiones simultáneas bastaban para tumbar el serve
# con un segfault (medido 17-jul: 1 petición OK, 2 concurrentes → caída y
# reinicio; ~105 s de recarga a 64k). Con Odysseus y Hermes apuntando al mismo
# endpoint, coincidir era cuestión de tiempo.
#
# La GPU procesa una petición cada vez de todas formas, así que serializar no
# cuesta rendimiento real: convierte un CRASH en una COLA. Es RLock por si algún
# camino llegara a anidar llamadas. En streaming se mantiene tomado durante todo
# el generador (si el cliente corta, GeneratorExit lo libera igual).
_INFER_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Carga del modelo (reutiliza lógica de _cmd_chat)
# ---------------------------------------------------------------------------

def _build_vision_handler():
    """
    Construye el chat handler de visión si `MOTOR_MMPROJ` apunta a un mmproj.

    El mmproj es el proyector de visión: convierte la imagen en embeddings que
    el MISMO GGUF de texto ya entiende. No hace falta servir los pesos completos
    ni otro modelo — solo este fichero (~175 MB para Gemma 4 12B).

    Devuelve None (modo texto) si no se pidió visión o si no se pudo montar.
    NUNCA lanza: un mmproj ausente o una llama-cpp sin soporte degradan a texto
    con aviso claro, en vez de tumbar el serve.

    Handler por familia (`MOTOR_MMPROJ_HANDLER`, def. auto por nombre de modelo).
    """
    import os
    mmproj = os.environ.get("MOTOR_MMPROJ", "").strip()
    if not mmproj:
        return None

    if not Path(mmproj).exists():
        print(f"[Server] AVISO: MOTOR_MMPROJ={mmproj} no existe → "
              f"arrancando en modo TEXTO (sin visión).")
        return None

    # llama-cpp NO valida el proyector al construir el handler (lo carga en la
    # primera imagen). Sin esta comprobación, un fichero truncado/erróneo se
    # anunciaba como "Visión ACTIVA" y /health mentía: cada imagen devolvía 500
    # ("Failed to load mtmd context"). Comprobar la cabecera es barato y ataja
    # el error real del operador (fichero a medio bajar o ruta equivocada).
    # Límite honesto: un GGUF válido PERO de otro modelo sí pasa este filtro y
    # fallará al inferir — ahí el mensaje de llama-cpp ya nombra el fichero.
    try:
        with open(mmproj, "rb") as fh:
            magic = fh.read(4)
    except OSError as exc:
        print(f"[Server] AVISO: no se pudo leer MOTOR_MMPROJ={mmproj} ({exc}) "
              f"→ modo TEXTO (sin visión).")
        return None
    if magic != b"GGUF":
        print(f"[Server] AVISO: {Path(mmproj).name} no es un GGUF válido "
              f"(cabecera {magic!r}) → modo TEXTO (sin visión). "
              f"¿Descarga incompleta o ruta equivocada?")
        return None

    handler_name = os.environ.get("MOTOR_MMPROJ_HANDLER", "Gemma4ChatHandler")
    try:
        from llama_cpp import llama_chat_format as _fmt
        HandlerCls = getattr(_fmt, handler_name)
    except (ImportError, AttributeError):
        print(f"[Server] AVISO: esta llama-cpp-python no tiene "
              f"'{handler_name}' → modo TEXTO (sin visión). "
              f"Actualiza llama-cpp-python o ajusta MOTOR_MMPROJ_HANDLER.")
        return None

    try:
        handler = HandlerCls(clip_model_path=mmproj, verbose=False)
    except Exception as exc:
        print(f"[Server] AVISO: no se pudo cargar el mmproj ({exc}) → "
              f"modo TEXTO (sin visión).")
        return None

    print(f"[Server] Visión ACTIVA: {handler_name} + {Path(mmproj).name}")
    return handler


def load_model(
    model_path: str,
    base_model: Optional[str] = None,
    cache_dir:  Optional[str] = None,
    api_key:    Optional[str] = None,
) -> None:
    """
    Carga el modelo en memoria. Se llama una sola vez al arrancar el servidor.
    Soporta tanto adapters LoRA / modelos HuggingFace como archivos GGUF.
    """
    import os

    _state.model_path = model_path
    # No machacar la API key en recargas (el watchdog llama sin api_key;
    # antes esto desactivaba la autenticación tras un hot-reload)
    if api_key is not None:
        _state.api_key = api_key

    # ── Modo GGUF (llama-cpp-python) ────────────────────────────────────────
    if model_path.lower().endswith(".gguf"):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "[Server] Para usar archivos .gguf instala llama-cpp-python:\n"
                "          pip install llama-cpp-python"
            )
        t0 = time.time()
        print(f"[Server] Cargando GGUF: {model_path}")
        from motor.hardware import detect_hardware
        _hw = detect_hardware()
        print(_hw)
        llama_kw = _hw.llama_kwargs()
        print(f"[Server] Parámetros llama-cpp: {llama_kw}")

        # ── Visión opcional (MOTOR_MMPROJ) ──────────────────────────────
        # El mismo GGUF de texto gana visión al cargarle el proyector
        # (mmproj) con el chat handler de la familia. Si algo falla, se
        # sigue en modo TEXTO con aviso: nunca romper el serve por esto.
        chat_handler = _build_vision_handler()
        if chat_handler is not None:
            llama_kw["chat_handler"] = chat_handler

        _state.llama_model = Llama(
            model_path = model_path,
            **llama_kw,
        )
        _state.is_gguf     = True
        _state.vision_ready = chat_handler is not None
        n_layers = llama_kw.get("n_gpu_layers", 0)
        _state.gpu_name    = _hw.gpu_name if _hw.cuda_available and n_layers != 0 else "CPU (GGUF)"
        _state.vram_total_gb = _hw.vram_total_gb
        _state.dtype_str   = "Q4_K_M"
        _state.load_time_s = round(time.time() - t0, 1)
        print(f"[Server] GGUF listo en {_state.load_time_s}s  "
              f"(perfil: {_hw.inference_profile})")
        return
    # ────────────────────────────────────────────────────────────────────────

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    _state.model_path = model_path
    _state.api_key    = api_key

    from pathlib import Path as _Path
    _state.is_adapter = (_Path(model_path) / "adapter_config.json").exists()

    print(f"[Server] Cargando {'adapter LoRA' if _state.is_adapter else 'modelo'}...")
    print(f"         Ruta: {model_path}")

    # Detectar GPU
    if torch.cuda.is_available():
        cap      = torch.cuda.get_device_capability()
        dtype    = torch.bfloat16 if cap[0] >= 8 else torch.float16
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"         GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    else:
        dtype   = torch.float32
        vram_gb = 0.0
        gpu_name = "CPU"
        print("         Modo CPU")

    _state.gpu_name       = gpu_name
    _state.vram_total_gb  = vram_gb
    _state.dtype_str      = "bf16" if dtype == torch.bfloat16 else (
        "fp16" if dtype == torch.float16 else "fp32"
    )

    # Cuantización 4-bit NF4 (helper compartido)
    use_4bit = vram_gb > 0 and vram_gb < 22
    if use_4bit:
        from motor._model_utils import apply_4bit_quantization
        model_kwargs = dict(device_map="auto")
        if not apply_4bit_quantization(model_kwargs, dtype=dtype, cpu_offload=True):
            use_4bit = False

    if not use_4bit:
        model_kwargs = dict(
            torch_dtype = dtype,
            device_map  = "auto" if vram_gb > 0 else "cpu",
        )
        print(f"         Modo: {_state.dtype_str}")

    # Resolver modelo base
    if _state.is_adapter:
        meta_path = Path(model_path) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"[Server] meta.json no encontrado en {model_path}.\n"
                "Usa --base-model para indicar el modelo base."
            )
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        _state.base_model_id = base_model or meta.get("model_id") or meta.get("base_model")
        if not _state.base_model_id:
            raise ValueError("[Server] No se pudo determinar el modelo base desde meta.json.")
        print(f"         Base: {_state.base_model_id}")
    else:
        _state.base_model_id = model_path

    # Cargar tokenizador
    _state.tokenizer = AutoTokenizer.from_pretrained(
        _state.base_model_id,
        cache_dir         = cache_dir,
        trust_remote_code = True,
    )
    if _state.tokenizer.pad_token is None:
        _state.tokenizer.pad_token = _state.tokenizer.eos_token

    # Cargar modelo base (con fallback a fp16 si bitsandbytes no está disponible)
    try:
        _state.model = AutoModelForCausalLM.from_pretrained(
            _state.base_model_id,
            cache_dir         = cache_dir,
            trust_remote_code = True,
            **model_kwargs,
        )
    except (ImportError, RuntimeError) as _bnb_err:
        if use_4bit and "bitsandbytes" in str(_bnb_err).lower():
            print(f"[Server] ⚠️  4-bit NF4 no disponible ({_bnb_err.__class__.__name__}).")
            print(f"[Server]    Reintentando en fp16 (Qwen2.5-3B cabe en 11 GB sin 4-bit)...")
            model_kwargs = dict(
                torch_dtype = dtype,
                device_map  = "auto" if vram_gb > 0 else "cpu",
            )
            _state.dtype_str = "fp16" if dtype == torch.float16 else "fp32"
            _state.model = AutoModelForCausalLM.from_pretrained(
                _state.base_model_id,
                cache_dir         = cache_dir,
                trust_remote_code = True,
                **model_kwargs,
            )
            print(f"[Server] Modelo cargado en fp16 (~6.7 GB VRAM).")
        else:
            raise

    # Cargar adapter si corresponde
    if _state.is_adapter:
        from peft import PeftModel
        print("[Server] Cargando adapter LoRA...")
        _state.model = PeftModel.from_pretrained(
            _state.model, model_path, is_trainable=False
        )

    _state.model.eval()
    _state.load_time_s = round(time.time() - t0, 1)
    print(f"[Server] Modelo listo en {_state.load_time_s}s")


# ---------------------------------------------------------------------------
# Lógica de inferencia (compartida por /chat y /chat/session)
# ---------------------------------------------------------------------------

def _infer(
    messages: List[dict],
    max_tokens:  int   = 512,
    temperature: float = 0.7,
    top_p:       float = 0.9,
    grammar_schema: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
):
    """
    Ejecuta inferencia sobre una lista de mensajes ChatML.
    Devuelve el texto generado (sin tokens especiales).
    Soporta tanto modelos HuggingFace como GGUF (llama-cpp-python).

    grammar_schema : dict | None
        JSON Schema que restringe la salida (solo modo GGUF).
        Cuando se pasa, llama-cpp fuerza matemáticamente al modelo a generar
        únicamente tokens compatibles con el schema → JSON siempre válido.
        En modo HuggingFace se ignora (no hay soporte de gramática).
    return_meta : bool
        Si True devuelve (texto, usage, finish_reason) en vez de solo texto.
        usage = {"prompt_tokens", "completion_tokens", "total_tokens"} con
        conteos reales (llama-cpp en GGUF, tokenizer en HF).
    """
    # ── Modo GGUF ────────────────────────────────────────────────────────────
    if _state.is_gguf:
        kw: dict = dict(
            messages          = messages,
            max_tokens        = max_tokens,
            temperature       = max(temperature, 0.01),
            top_p             = top_p,
            repeat_penalty    = 1.1,
        )
        if grammar_schema is not None:
            # Construir gramática desde JSON Schema.
            # LlamaGrammar.from_json_schema() está disponible en llama-cpp-python ≥ 0.2.26.
            # Si la versión instalada no lo soporta, degradamos silenciosamente.
            try:
                from llama_cpp import LlamaGrammar
                kw["grammar"] = LlamaGrammar.from_json_schema(
                    json.dumps(grammar_schema)
                )
            except Exception:
                pass  # gramática no disponible → inferencia sin restricción
        with _INFER_LOCK:      # llama-cpp no es thread-safe (ver _INFER_LOCK)
            result = _state.llama_model.create_chat_completion(**kw)
        choice = result["choices"][0]
        content = _strip_gemma_channels(choice["message"]["content"] or "")
        if return_meta:
            usage = result.get("usage") or {}
            return content, usage, choice.get("finish_reason") or "stop"
        return content
    # ────────────────────────────────────────────────────────────────────────

    import torch

    text = _state.tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,
    )
    inputs = _state.tokenizer(text, return_tensors="pt").to(_state.model.device)

    with torch.no_grad(), _INFER_LOCK:   # serializado igual que el camino GGUF
        output = _state.model.generate(
            **inputs,
            max_new_tokens     = max_tokens,
            do_sample          = temperature > 0,
            temperature        = temperature if temperature > 0 else 1.0,
            top_p              = top_p,
            repetition_penalty = 1.1,
            pad_token_id       = _state.tokenizer.eos_token_id,
        )

    n_prompt = inputs["input_ids"].shape[1]
    decoded = _state.tokenizer.decode(
        output[0][n_prompt:],
        skip_special_tokens = True,
    ).strip()
    if return_meta:
        n_completion = output[0].shape[0] - n_prompt
        usage = {
            "prompt_tokens":     int(n_prompt),
            "completion_tokens": int(n_completion),
            "total_tokens":      int(n_prompt + n_completion),
        }
        return decoded, usage, "stop"
    return decoded


# ---------------------------------------------------------------------------
# Log de interacciones (S10.1) — compartido por /chat/session y /v1/chat/completions
# ---------------------------------------------------------------------------

def _log_interaction(
    interaction_id: str,
    user_msg: str,
    assistant: str,
    ms: int,
    session_id: Optional[str] = None,
    turn: Optional[int] = None,
    endpoint: str = "chat/session",
    extra: Optional[dict] = None,
) -> None:
    """
    Añade una interacción al log JSONL para continual learning y DPO.

    Esquema consumido por `learn --auto` y DPOBuilder: requieren `user_msg`,
    `assistant` y `feedback` (este último se rellena después vía POST /feedback
    usando `interaction_id` como clave). Nunca lanza: el log no debe bloquear
    ni romper una respuesta.
    """
    if not _state.interaction_log_path:
        return
    try:
        import datetime as _dt
        log_path = Path(_state.interaction_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotación por tamaño (default 50 MB, MOTOR_LOG_MAX_MB para cambiar):
        # learn --auto y DPOBuilder releen el archivo entero, y un JSONL
        # append-only infinito degrada el ciclo CL nocturno.
        try:
            max_mb = float(os.getenv("MOTOR_LOG_MAX_MB", "50"))
            if log_path.exists() and log_path.stat().st_size > max_mb * 1024 * 1024:
                stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d%H%M%S")
                log_path.rename(log_path.with_suffix(f".{stamp}.jsonl"))
        except Exception:
            pass  # la rotación nunca bloquea el logging
        _now = _dt.datetime.now(_dt.timezone.utc)
        log_entry = {
            "id":         interaction_id,
            "timestamp":  _now.isoformat().replace("+00:00", "Z"),
            "session_id": session_id,
            "turn":       turn,
            "user_msg":   user_msg,
            "assistant":  assistant,
            "model":      str(Path(_state.model_path).name),
            "ms":         ms,
            "endpoint":   endpoint,
            "feedback":   None,   # se rellena via POST /feedback
        }
        if extra:
            log_entry.update(extra)
        with open(log_path, "a", encoding="utf-8") as _lf:
            _lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # el log nunca bloquea la respuesta


# ---------------------------------------------------------------------------
# Schemas Pydantic (nivel de módulo — necesario para que FastAPI los reconozca
# como request body y no como query parameters)
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel as _BaseModel
    _pydantic_import_error: Optional[str] = None

    class ChatRequest(_BaseModel):
        message:     str
        system:      Optional[str]  = "Eres un asistente útil. Responde de forma clara y concisa."
        max_tokens:  int            = 512
        temperature: float          = 0.7
        top_p:       float          = 0.9

    class SessionChatRequest(_BaseModel):
        message:     str
        session_id:  Optional[str]  = None
        system:      Optional[str]  = "Eres un asistente útil. Responde de forma clara y concisa."
        max_tokens:  int            = 512
        temperature: float          = 0.7
        top_p:       float          = 0.9

    class ChatResponse(_BaseModel):
        response:   str
        model:      str
        ms:         int

    class SessionChatResponse(_BaseModel):
        response:   str
        session_id: str
        turn:       int
        model:      str
        ms:         int

    class HealthResponse(_BaseModel):
        status:        str
        model:         str
        base_model:    str
        is_adapter:    bool
        gpu:           str
        vram_total_gb: float
        vram_used_gb:  float
        dtype:         str
        load_time_s:   float
        uptime_s:      float
        # True si el GGUF se cargó con mmproj → acepta imágenes (visión).
        vision:        bool = False
        hardware:      Optional[Dict[str, Any]] = None

    class AgentRequest(_BaseModel):
        task:      str
        max_steps: int = 12

    class AgentResponse(_BaseModel):
        answer:      str
        steps:       list
        steps_taken: int
        success:     bool
        ms:          int

    class FeedbackRequest(_BaseModel):
        interaction_id: str
        rating:         int          # 1 = 👍, -1 = 👎
        comment:        Optional[str] = None

except ImportError as e:
    # Pydantic no instalado o versión incompatible — guardamos el error
    # para lanzarlo con un mensaje claro cuando se intente crear la app
    _pydantic_import_error = str(e)


# ── OpenAI-compatible Pydantic models (nivel de módulo) ─────────────
# DEBEN estar a nivel de módulo, no dentro de create_app(),
# porque from __future__ import annotations convierte las anotaciones
# en strings y FastAPI no puede resolverlas desde el scope local.
try:
    from pydantic import BaseModel as _OAIBase, Field as _OAIField
except ImportError:
    _OAIBase  = object  # type: ignore
    _OAIField = lambda *a, **kw: None  # type: ignore

class _OAIMessage(_OAIBase):  # type: ignore
    role:    str
    # str | lista de partes [{"type": "text", "text": ...}] | None — los
    # clientes OpenAI modernos (Odysseus incluido) pueden enviar content
    # como lista de partes; con `content: str` eso devolvía 422.
    content: Any = ""
    # Tool-calling (OpenAI): assistant → tool_calls; role="tool" → tool_call_id.
    # Sin estos campos el historial de un loop agéntico devolvía 422.
    tool_calls:   Optional[List[dict]] = None
    tool_call_id: Optional[str]        = None
    name:         Optional[str]        = None


# Sintaxis nativa de tool call de Gemma 4 (observada en vivo, 10-jun-2026):
#   <|tool_call>call:file_organize{path:<|"|>Descargas<|"|>}<tool_call|>
# llama-cpp-python no la convierte a la estructura OpenAI (sus parsers cubren
# functionary/chatml), así que el modelo "llama" a la herramienta pero el
# cliente recibe texto plano. Este parser tolerante cierra ese hueco.
_GEMMA_TOOL_RE = re.compile(
    r"<\|?/?tool_call\|?>\s*call:\s*([A-Za-z0-9_.\-]+)\s*\{(.*?)\}\s*<\|?/?tool_call\|?>",
    re.DOTALL,
)


# Canales de Gemma 4: con thinking desactivado el modelo igualmente emite
#   <|channel>thought\n<channel|>[respuesta final]
# (model card oficial). llama-cpp no siempre los filtra → los quitamos.
_GEMMA_CHANNEL_RE = re.compile(r"^\s*<\|channel>thought\n(.*?)<channel\|>\s*", re.DOTALL)
_GEMMA_CHANNEL_MARK = "<|channel>thought"


def _strip_gemma_channels(text: str) -> str:
    """Elimina el bloque de canal 'thought' inicial de una respuesta Gemma."""
    return _GEMMA_CHANNEL_RE.sub("", text or "", count=1)


def _gemma_stream_hold(buf: str):
    """
    Decide qué hacer con el buffer inicial de un streaming que podría empezar
    con el canal 'thought' de Gemma. Devuelve (decidido, texto_a_emitir):
      - (False, "")   → indeciso o dentro del canal: seguir reteniendo
      - (True, texto) → emitir `texto` y dejar de retener
    """
    s = buf.lstrip()
    if len(s) < len(_GEMMA_CHANNEL_MARK):
        if _GEMMA_CHANNEL_MARK.startswith(s):
            return False, ""          # podría ser el canal → retener
        return True, buf              # claramente no es el canal
    if s.startswith(_GEMMA_CHANNEL_MARK):
        if "<channel|>" in s:
            return True, _strip_gemma_channels(buf)
        if len(s) > 4000:
            return True, buf          # cierre que no llega: no retener más
        return False, ""
    return True, buf


def _parse_gemma_tool_calls(text: str) -> Optional[List[dict]]:
    """
    Extrae tool calls en sintaxis nativa Gemma de un texto y las devuelve
    en formato OpenAI ([{id, type, function: {name, arguments}}]).
    Devuelve None si el texto no contiene ninguna.
    """
    matches = _GEMMA_TOOL_RE.findall(text or "")
    if not matches:
        return None
    calls: List[dict] = []
    for name, raw_args in matches:
        args = raw_args.replace('<|"|>', '"').strip()
        # Claves sin comillas → JSON válido: {path: "x"} → {"path": "x"}
        args = re.sub(r'([{,[]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', "{" + args + "}")
        parsed = None
        for candidate in (
            args,
            # Rutas Windows: C:\Users\... son escapes JSON inválidos (\U) →
            # segundo intento con los backslashes doblados
            args.replace("\\", "\\\\"),
        ):
            try:
                json.loads(candidate)
                parsed = candidate
                break
            except Exception:
                continue
        if parsed is None:
            # Argumentos no parseables: se entregan crudos para que el
            # cliente decida (mejor que perder la llamada)
            parsed = json.dumps({"raw": raw_args})
        calls.append({
            "id":   f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": parsed},
        })
    return calls


def _oai_content_to_text(content: Any) -> str:
    """Normaliza el campo content de un mensaje OpenAI a texto plano."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


# Tipos de parte que transportan una imagen en el formato OpenAI.
_IMAGE_PART_TYPES = ("image_url", "input_image", "image")


def _has_image_part(content: Any) -> bool:
    """True si el content trae alguna parte de imagen."""
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES
        for p in content
    )


def _oai_content_for_model(content: Any) -> Any:
    """Content tal y como debe llegar al modelo.

    Con visión activa, las partes de imagen se PRESERVAN en formato OpenAI (el
    chat handler de llama-cpp las consume así). Sin visión, o sin imágenes, se
    aplana a texto como siempre — `_oai_content_to_text` descarta las imágenes,
    por eso el endpoint rechaza antes las peticiones con imagen sin visión.
    """
    if _state.vision_ready and _has_image_part(content):
        return content
    return _oai_content_to_text(content)


def _content_for_log(content: Any) -> str:
    """Aplana content a texto para el LOG y los conteos.

    Las imágenes se anotan como '[imagen]', nunca su base64: el log canónico
    (Regla 11) lo releen learn/DPO, y un data-URI de megabytes lo envenenaría
    y dispararía la rotación del fichero.
    """
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES:
                parts.append("[imagen]")
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return _oai_content_to_text(content)


# Marcadores de desbordamiento de contexto. llama-cpp NO usa un tipo de
# excepción propio ni un mensaje estable: hay que reconocerlo por texto, y el
# texto CAMBIA entre versiones y caminos de código. Vistos en vivo:
#   - "Requested tokens (40108) exceed context window of 4096"  (llama-cpp viejo)
#   - "Prompt exceeds n_ctx: 40108 > 16384"                     (0.3.28 / chat handler)
# Una lista con ambos evita que el fix se muera en silencio al actualizar.
_CTX_OVERFLOW_MARKERS = (
    "exceed context window",
    "exceeds n_ctx",
    "exceed n_ctx",
)


def _is_context_overflow(exc: Exception) -> bool:
    """True si la excepción es un desbordamiento del contexto del modelo."""
    msg = str(exc).lower()
    return any(m in msg for m in _CTX_OVERFLOW_MARKERS)


# Marcadores de imagen inválida — el cliente mandó un data-URI/URL que no se
# pudo decodificar. Verificados en vivo (17-jul): base64 corrupto, data-URI
# truncado, bytes que no son imagen, URL desconocida/inalcanzable, image_url
# malformado. Es culpa del cliente → 400, no un 500 opaco del servidor.
_IMAGE_ERROR_MARKERS = (
    "incorrect padding",
    "invalid base64",
    "failed to create bitmap",
    "cannot identify image",
    "unknown url type",
    "urlopen error",
    "connection refused",
    "replace() argument 1 must be str",   # image_url dict sin 'url'
)


def _is_image_error(exc: Exception) -> bool:
    """True si la excepción viene de un data-URI/URL de imagen inválido."""
    msg = str(exc).lower()
    return any(m in msg for m in _IMAGE_ERROR_MARKERS)


def _raise_inference_error(exc: Exception, context: str = "Error en inferencia"):
    """Traduce una excepción de inferencia a HTTPException con el código
    adecuado. Desbordamiento de contexto e imagen inválida son culpa del
    cliente → 400 (con un `type` accionable) en vez de un 500 opaco. El resto
    siguen siendo 500. Ver `_CTX_OVERFLOW_MARKERS` / `_IMAGE_ERROR_MARKERS`."""
    from fastapi import HTTPException
    if _is_context_overflow(exc):
        raise HTTPException(status_code=400, detail=(
            f"context_length_exceeded: {exc}. El prompt supera el contexto del "
            f"modelo — reduce el historial/documentos, o sirve con MOTOR_N_CTX "
            f"mayor (ver motor/hardware.py)."))
    if _is_image_error(exc):
        raise HTTPException(status_code=400, detail={
            "message": (f"invalid_image: no se pudo decodificar la imagen ({exc}). "
                        f"Envía un data-URI base64 válido (data:image/...;base64,...) "
                        f"o una URL de imagen accesible."),
            "type": "invalid_image",
        })
    raise HTTPException(status_code=500, detail=f"{context}: {exc}")


def _oai_conversation_key(messages: list) -> tuple:
    """Deriva (session_id, turn) para una petición /v1 sin estado.

    El endpoint /v1 es stateless: cada llamada trae toda la conversación en
    `messages`, pero nosotros solo logueamos el último turno. Sin enlace, el
    pase de reflexión no puede leer la reacción del usuario entre turnos. Aquí
    reconstruimos ese enlace: el fingerprint de la conversación es el hash del
    PRIMER mensaje de usuario (estable entre turnos de la misma charla), y el
    turno es el nº de mensajes de usuario en la petición. No es perfecto (dos
    conversaciones que empiezan idénticas colisionan; editar el historial lo
    rompe), pero enlaza el caso común de Odysseus/Hermes.
    """
    import hashlib

    def _mget(m, k):
        return m.get(k) if isinstance(m, dict) else getattr(m, k, None)

    user_texts = [
        _oai_content_to_text(_mget(m, "content"))
        for m in messages if _mget(m, "role") == "user"
    ]
    user_texts = [t for t in user_texts if t and t.strip()]
    if not user_texts:
        return None, None
    first = user_texts[0].strip()[:400]
    sid = "oai-" + hashlib.sha1(first.encode("utf-8")).hexdigest()[:12]
    return sid, len(user_texts)


class _OAIRequest(_OAIBase):  # type: ignore
    # Optional: el Deep Research de Odysseus sondea con model=null y el 422
    # resultante mataba la investigación. Solo servimos un modelo: se ignora.
    model:       Optional[str]  = ""
    messages:    List[_OAIMessage]
    # None o 0 = sin especificar → default generoso (2048). El antiguo
    # default 512 cortaba a media frase las respuestas largas de Gemma 4
    # (visto en vivo con Odysseus: finish_reason=length a los 512).
    max_tokens:  Optional[int]  = None
    temperature: float          = 0.7
    top_p:       float          = 0.9
    stream:      bool           = False
    # Tool-calling nativo (Gemma 4 / llama-cpp): antes Pydantic descartaba
    # estos campos en silencio y el modelo nunca veía las herramientas.
    tools:       Optional[List[dict]] = None
    tool_choice: Optional[Any]        = None   # "auto" | "none" | {"type": "function", ...}

class _OAIChoice(_OAIBase):  # type: ignore
    index:        int           = 0
    message:      _OAIMessage   = _OAIField(default_factory=lambda: _OAIMessage(role="assistant", content=""))  # type: ignore
    finish_reason: str          = "stop"

class _OAIUsage(_OAIBase):  # type: ignore
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0

class _OAIResponse(_OAIBase):  # type: ignore
    id:      str  = "chatcmpl-0"
    object:  str  = "chat.completion"
    created: int  = 0
    model:   str  = ""
    choices: List[_OAIChoice] = []
    usage:   _OAIUsage        = _OAIField(default_factory=_OAIUsage)  # type: ignore

class _OAIModelEntry(_OAIBase):  # type: ignore
    id:       str  = ""
    object:   str  = "model"
    owned_by: str  = "motor-de-loras"

class _OAIModelList(_OAIBase):  # type: ignore
    object: str = "list"
    data:   List[_OAIModelEntry] = []


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

def create_app() -> "fastapi.FastAPI":
    """
    Crea y devuelve la aplicación FastAPI con todos los endpoints.
    El modelo ya debe estar cargado en _state antes de llamar a esto.
    """
    if _pydantic_import_error:
        raise ImportError(
            f"Pydantic no disponible o version incompatible: {_pydantic_import_error}\n"
            f"Instala pydantic con: pip install pydantic"
        )

    try:
        from fastapi import FastAPI, HTTPException, Depends, Header
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError(
            "[Server] FastAPI no instalado.\n"
            "Ejecuta: pip install fastapi uvicorn"
        )

    if not _state.is_gguf:
        import torch

    app = FastAPI(
        title       = "Fábrica de LoRAs — API",
        description = "Servidor REST para inferencia con adapters LoRA especializados.",
        version     = "1.0.0",
    )

    # CORS: configurable vía MOTOR_CORS_ORIGINS (lista separada por comas).
    # Default "*" para no romper clientes existentes; restringir en producción.
    _cors_origins = [
        o.strip() for o in os.getenv("MOTOR_CORS_ORIGINS", "*").split(",") if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins  = _cors_origins,
        allow_methods  = ["GET", "POST", "DELETE"],
        allow_headers  = ["*"],
    )

    # --- Dependencia de autenticación (opcional) ---
    def _check_api_key(authorization: Optional[str] = Header(default=None)):
        if _state.api_key is None:
            return  # Sin protección
        if authorization != f"Bearer {_state.api_key}":
            raise HTTPException(status_code=401, detail="API key inválida o ausente.")

    # --- Autenticación OBLIGATORIA (endpoints que ejecutan acciones reales) ---
    def _require_api_key(authorization: Optional[str] = Header(default=None)):
        """
        Para /agent: ejecuta herramientas con efectos en el host (mover
        archivos, lanzar procesos de la whitelist). Sin API key configurada,
        cualquier dispositivo de la LAN podría ordenar acciones → se exige
        que el servidor tenga clave Y que la petición la traiga.
        """
        if _state.api_key is None:
            raise HTTPException(
                status_code=403,
                detail="POST /agent deshabilitado: el agente ejecuta herramientas "
                       "reales (archivos, procesos). Arranca el servidor con "
                       "--api-key para habilitarlo.",
            )
        if authorization != f"Bearer {_state.api_key}":
            raise HTTPException(status_code=401, detail="API key inválida o ausente.")

    _start_time = time.time()

    # ----------------------------------------------------------------
    # GET /  →  Chat UI (página HTML)
    # ----------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def chat_ui():
        from fastapi.responses import HTMLResponse
        model_name = str(Path(_state.model_path).name)
        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fábrica de LoRAs · {model_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117; color: #e0e0e0; height: 100vh;
      display: flex; flex-direction: column;
    }}
    header {{
      background: #1a1d27; border-bottom: 1px solid #2a2d3a;
      padding: 12px 24px; display: flex; align-items: center; gap: 12px;
    }}
    .model-badge {{
      background: #6c47ff; color: white; font-size: 12px;
      padding: 3px 10px; border-radius: 12px; font-weight: 600;
    }}
    .status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #22c55e; flex-shrink: 0; }}
    header h1 {{ font-size: 15px; font-weight: 600; flex: 1; }}
    #health-info {{ font-size: 12px; color: #888; }}

    /* Tabs */
    .tabs {{
      display: flex; background: #1a1d27;
      border-bottom: 1px solid #2a2d3a; padding: 0 24px;
    }}
    .tab {{
      padding: 10px 20px; font-size: 13px; cursor: pointer;
      border-bottom: 2px solid transparent; color: #888;
      transition: color 0.15s, border-color 0.15s;
    }}
    .tab.active {{ color: #e0e0e0; border-bottom-color: #6c47ff; }}
    .tab:hover:not(.active) {{ color: #c0c0c0; }}

    /* Panels */
    .panel {{ display: none; flex: 1; flex-direction: column; overflow: hidden; }}
    .panel.active {{ display: flex; }}

    /* ── Chat panel ── */
    #system-row {{
      padding: 8px 24px; display: flex; align-items: center; gap: 8px;
      background: #161920; border-bottom: 1px solid #2a2d3a;
    }}
    #system-row label {{ font-size: 12px; color: #888; white-space: nowrap; }}
    #system-input {{
      flex: 1; background: #0f1117; border: 1px solid #2a2d3a;
      border-radius: 8px; padding: 5px 10px; color: #e0e0e0;
      font-size: 12px; font-family: inherit;
    }}
    #messages {{
      flex: 1; overflow-y: auto; padding: 20px 24px;
      display: flex; flex-direction: column; gap: 14px;
    }}
    .msg {{ display: flex; gap: 10px; max-width: 820px; }}
    .msg.user {{ align-self: flex-end; flex-direction: row-reverse; }}
    .avatar {{
      width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center; font-size: 15px;
    }}
    .avatar.ai {{ background: #6c47ff; }}
    .avatar.user {{ background: #2563eb; }}
    .bubble {{
      background: #1e2130; border: 1px solid #2a2d3a;
      border-radius: 14px; padding: 10px 14px; font-size: 14px;
      line-height: 1.6; max-width: 680px; white-space: pre-wrap;
    }}
    .msg.user .bubble {{ background: #1e3a5f; border-color: #2a4a7a; }}
    .bubble.thinking {{ color: #888; font-style: italic; }}
    #input-area {{
      background: #1a1d27; border-top: 1px solid #2a2d3a;
      padding: 14px 24px; display: flex; gap: 10px; align-items: flex-end;
    }}
    #msg-input {{
      flex: 1; background: #0f1117; border: 1px solid #2a2d3a;
      border-radius: 12px; padding: 10px 14px; color: #e0e0e0;
      font-size: 14px; font-family: inherit; resize: none;
      min-height: 46px; max-height: 180px; line-height: 1.5;
    }}
    #msg-input:focus, #system-input:focus, #task-input:focus {{
      outline: none; border-color: #6c47ff;
    }}
    button {{
      background: #6c47ff; color: white; border: none; border-radius: 10px;
      padding: 10px 18px; font-size: 13px; cursor: pointer; font-weight: 600;
      transition: background 0.15s; white-space: nowrap;
    }}
    button:hover {{ background: #5a38d9; }}
    button:disabled {{ background: #3a3d4a; cursor: not-allowed; }}
    .btn-ghost {{
      background: transparent !important; border: 1px solid #2a2d3a;
      color: #888 !important; padding: 6px 12px !important; font-size: 12px !important;
    }}
    .btn-ghost:hover {{ border-color: #6c47ff !important; color: #e0e0e0 !important; }}

    /* ── Agent panel ── */
    #agent-panel {{ padding: 24px; gap: 16px; overflow-y: auto; }}
    #task-input {{
      width: 100%; background: #1e2130; border: 1px solid #2a2d3a;
      border-radius: 10px; padding: 12px 14px; color: #e0e0e0;
      font-size: 14px; font-family: inherit; resize: vertical;
      min-height: 80px; line-height: 1.5;
    }}
    .agent-controls {{ display: flex; gap: 10px; align-items: center; }}
    .agent-controls label {{ font-size: 12px; color: #888; }}
    .agent-controls input[type=number] {{
      width: 60px; background: #0f1117; border: 1px solid #2a2d3a;
      border-radius: 6px; padding: 5px 8px; color: #e0e0e0; font-size: 13px;
    }}
    #agent-result {{
      background: #1e2130; border: 1px solid #2a2d3a;
      border-radius: 10px; padding: 16px;
    }}
    #agent-result h3 {{ font-size: 13px; color: #888; margin-bottom: 10px; }}
    #agent-answer {{
      font-size: 15px; line-height: 1.7; white-space: pre-wrap;
      border-bottom: 1px solid #2a2d3a; padding-bottom: 14px; margin-bottom: 14px;
    }}
    #agent-steps {{ display: flex; flex-direction: column; gap: 10px; }}
    .step {{
      border: 1px solid #2a2d3a; border-radius: 8px; overflow: hidden;
    }}
    .step-header {{
      background: #161920; padding: 8px 14px; font-size: 12px;
      cursor: pointer; display: flex; justify-content: space-between;
      color: #aaa; user-select: none;
    }}
    .step-header:hover {{ background: #1e2130; }}
    .step-body {{
      padding: 10px 14px; font-size: 13px; line-height: 1.6;
      white-space: pre-wrap; display: none;
    }}
    .step-body.open {{ display: block; }}
    .step-body .label {{ color: #6c47ff; font-weight: 600; font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }}
    .step-body .value {{ margin-bottom: 10px; }}
    .step-body .obs {{ background: #0f1117; border-radius: 6px; padding: 8px 10px; font-family: monospace; font-size: 12px; }}
    .badge-action {{ background: #1e3a5f; color: #60a5fa; padding: 1px 7px; border-radius: 10px; font-size: 11px; }}
    .badge-final {{ background: #14532d; color: #4ade80; padding: 1px 7px; border-radius: 10px; font-size: 11px; }}
    #agent-placeholder {{ color: #555; font-size: 14px; text-align: center; padding: 40px; }}

    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: #2a2d3a; border-radius: 3px; }}

    /* ── Feedback buttons (S10.1) ── */
    .fb-row {{ display: flex; gap: 6px; margin-top: 4px; }}
    .fb-btn {{
      background: transparent; border: 1px solid #2a2d3a; border-radius: 6px;
      color: #888; padding: 2px 8px; font-size: 13px; cursor: pointer;
      transition: border-color 0.15s, color 0.15s;
    }}
    .fb-btn:hover {{ border-color: #6c47ff; color: #e0e0e0; background: transparent; }}

    /* ── Digestor panel ── */
    #digestor-panel {{
      padding: 24px; gap: 20px; overflow-y: auto; display: flex;
      flex-direction: column; max-width: 900px;
    }}
    #digestor-panel label {{
      font-size: 13px; color: #aaa; margin-bottom: 4px; font-weight: 600;
    }}
    #digestor-panel textarea, #digestor-panel input[type=text], #digestor-panel select {{
      background: #1e2130; border: 1px solid #2a2d3a;
      border-radius: 10px; padding: 10px 14px; color: #e0e0e0;
      font-size: 14px; font-family: inherit; width: 100%;
    }}
    #digestor-panel textarea {{
      resize: vertical; min-height: 60px; line-height: 1.5;
    }}
    #digestor-panel select {{
      cursor: pointer; appearance: none;
      background-image: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><path fill='gray' d='M4 6l4 4 4-4'/></svg>");
      background-repeat: no-repeat; background-position: right 10px center;
      padding-right: 30px;
    }}
    .file-drop {{
      border: 2px dashed #2a2d3a; border-radius: 12px;
      padding: 40px 20px; text-align: center; cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
    }}
    .file-drop:hover, .file-drop.drag-over {{
      border-color: #6c47ff; background: #1a1d30;
    }}
    .file-drop .icon {{ font-size: 32px; margin-bottom: 8px; }}
    .file-drop .text {{ font-size: 13px; color: #888; }}
    .file-drop .file-name {{ font-size: 14px; color: #c0c0c0; font-weight: 600; margin-top: 6px; }}
    #file-input {{ display: none; }}
    .form-row {{ display: flex; gap: 12px; }}
    .form-row > div {{ flex: 1; }}
    .semaforo-badge {{
      display: inline-block; padding: 4px 14px; border-radius: 14px;
      font-weight: 700; font-size: 14px; letter-spacing: 1px;
    }}
    .semaforo-ROJO    {{ background: #3b1111; color: #f87171; border: 1px solid #7f1d1d; }}
    .semaforo-AMARILLO {{ background: #3b2f11; color: #fbbf24; border: 1px solid #7f6d1d; }}
    .semaforo-VERDE   {{ background: #113b1a; color: #4ade80; border: 1px solid #1d7f2a; }}
    #digestor-result {{
      background: #1e2130; border: 1px solid #2a2d3a;
      border-radius: 10px; padding: 20px; display: none;
    }}
    #digestor-result h3 {{ font-size: 14px; color: #888; margin-bottom: 12px; }}
    .metric-row {{
      display: flex; gap: 8px; margin-bottom: 6px; font-size: 13px;
    }}
    .metric-label {{ color: #888; min-width: 160px; }}
    .metric-value {{ color: #e0e0e0; font-weight: 600; }}
    .warning-item {{
      background: #2a2511; border: 1px solid #5a4a1a;
      border-radius: 6px; padding: 6px 10px; margin-top: 4px;
      font-size: 12px; color: #fbbf24;
    }}
    .download-link {{
      display: inline-block; margin-top: 14px;
      background: #6c47ff; color: white; text-decoration: none;
      padding: 8px 18px; border-radius: 8px; font-size: 13px; font-weight: 600;
    }}
    .download-link:hover {{ background: #5a38d9; }}
  </style>
</head>
<body>
  <header>
    <div class="status-dot" id="dot"></div>
    <h1>Fábrica de LoRAs</h1>
    <span class="model-badge">{model_name}</span>
    <span id="health-info">cargando...</span>
  </header>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('chat')">💬 Chat</div>
    <div class="tab" onclick="switchTab('agent')">🤖 Agente</div>
    <div class="tab" onclick="switchTab('digestor')">📄 Digestor</div>
  </div>

  <!-- ══ Chat panel ══ -->
  <div class="panel active" id="panel-chat">
    <div id="system-row">
      <label>System prompt:</label>
      <input id="system-input" type="text"
        value="Eres un asistente útil. Responde de forma clara y concisa.">
      <button class="btn-ghost" onclick="clearSession()">Nueva sesión</button>
    </div>
    <div id="messages"></div>
    <div id="input-area">
      <textarea id="msg-input"
        placeholder="Escribe tu mensaje… (Enter para enviar, Shift+Enter para nueva línea)"
        rows="1" onkeydown="handleKey(event)"></textarea>
      <button id="send-btn" onclick="sendMessage()">Enviar</button>
    </div>
  </div>

  <!-- ══ Agent panel ══ -->
  <div class="panel" id="panel-agent">
    <div id="agent-panel">
      <textarea id="task-input"
        placeholder="Describe la tarea que quieres que resuelva el agente…&#10;Ejemplo: ¿Cuántos adapters hay entrenados? ¿Cuánta VRAM libre hay? Resume el último log de entrenamiento."></textarea>
      <div class="agent-controls">
        <button id="agent-btn" onclick="runAgent()">▶ Ejecutar agente</button>
        <label>Pasos máx:</label>
        <input type="number" id="max-steps" value="12" min="2" max="20">
      </div>
      <div id="agent-result" style="display:none">
        <h3 id="agent-result-title">Respuesta</h3>
        <div id="agent-answer"></div>
        <div id="agent-steps"></div>
      </div>
      <div id="agent-placeholder">
        El agente puede leer archivos, listar directorios, ejecutar comandos
        de solo lectura y hacer peticiones HTTP para resolver tareas complejas.
      </div>
    </div>
  </div>

  <!-- ══ Digestor panel ══ -->
  <div class="panel" id="panel-digestor">
    <div id="digestor-panel">
      <div>
        <label>📂 Archivo de datos</label>
        <div class="file-drop" id="file-drop" onclick="document.getElementById('file-input').click()">
          <div class="icon">📁</div>
          <div class="text">Haz clic o arrastra un archivo aquí<br><small>CSV, JSON, TXT, PDF, DOCX, HTML, MP3, MP4...</small></div>
          <div class="file-name" id="file-name"></div>
        </div>
        <input type="file" id="file-input" onchange="fileSelected(event)">
      </div>
      <div>
        <label>🎯 Tarea para el LLM</label>
        <textarea id="digestor-task" placeholder="Ej: ¿Sobrevivió este pasajero al Titanic? Responde YES o NO."></textarea>
      </div>
      <div class="form-row">
        <div>
          <label>🏷️ Columna de etiqueta (opcional)</label>
          <input type="text" id="digestor-label-col" placeholder="Ej: Survived">
        </div>
        <div>
          <label>🗺️ Label map (opcional)</label>
          <input type="text" id="digestor-label-map" placeholder="Ej: 0:NO,1:YES">
        </div>
        <div>
          <label>📦 Formato salida</label>
          <select id="digestor-format">
            <option value="chatml">ChatML (recomendado)</option>
            <option value="unsloth">Unsloth (Alpaca)</option>
            <option value="llamafactory">LLaMA-Factory</option>
            <option value="axolotl">Axolotl</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div>
          <label>🏥 Dominio (opcional)</label>
          <select id="digestor-domain">
            <option value="auto">Auto-detectar</option>
            <option value="general">General</option>
            <option value="medical">Médico</option>
            <option value="legal">Legal</option>
            <option value="financial">Financiero</option>
            <option value="technical">Técnico</option>
          </select>
        </div>
        <div>
          <label>🤖 Modelo HF ID (opcional)</label>
          <input type="text" id="digestor-model-id" placeholder="Ej: Qwen/Qwen2.5-7B-Instruct">
        </div>
      </div>
      <button id="digestor-btn" onclick="runDigestor()">🔄 Procesar datos</button>
      <div id="digestor-result">
        <h3 id="digestor-result-title">Resultado</h3>
        <div id="digestor-metrics"></div>
        <div id="digestor-warnings"></div>
        <a class="download-link" id="digestor-download" href="#" download>⬇ Descargar dataset</a>
      </div>
    </div>
  </div>

<script>
  let sessionId = null;
  let activeTab = 'chat';

  // ── Tabs ─────────────────────────────────────────────────────────
  function switchTab(tab) {{
    activeTab = tab;
    document.querySelectorAll('.tab').forEach(t => {{
      t.classList.toggle('active', t.textContent.toLowerCase().includes(tab));
    }});
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.getElementById('panel-' + tab).classList.add('active');
  }}

  // ── Health ───────────────────────────────────────────────────────
  async function fetchHealth() {{
    try {{
      const r = await fetch('/health');
      const d = await r.json();
      document.getElementById('health-info').textContent =
        `${{d.gpu}} · ${{d.vram_used_gb}}/${{d.vram_total_gb}} GB · ${{d.dtype}}`;
      document.getElementById('dot').style.background = '#22c55e';
    }} catch(e) {{
      document.getElementById('health-info').textContent = 'sin conexión';
      document.getElementById('dot').style.background = '#ef4444';
    }}
  }}

  // ── Chat ─────────────────────────────────────────────────────────
  function addMsg(role, text, thinking=false) {{
    const wrap = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'msg ' + (role === 'user' ? 'user' : 'ai');
    div.innerHTML = `
      <div class="avatar ${{role === 'user' ? 'user' : 'ai'}}">
        ${{role === 'user' ? '👤' : '🤖'}}
      </div>
      <div class="bubble ${{thinking ? 'thinking' : ''}}">${{esc(text)}}</div>`;
    wrap.appendChild(div);
    wrap.scrollTop = wrap.scrollHeight;
    return div.querySelector('.bubble');
  }}

  function esc(t) {{
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }}

  async function sendMessage() {{
    const input = document.getElementById('msg-input');
    const msg = input.value.trim();
    if (!msg) return;
    const btn = document.getElementById('send-btn');
    btn.disabled = true;
    input.value = ''; input.style.height = 'auto';
    addMsg('user', msg);
    const thinking = addMsg('ai', 'Pensando…', true);
    const system = document.getElementById('system-input').value;
    const body = {{ message: msg, system, max_tokens: 512 }};
    if (sessionId) body.session_id = sessionId;
    try {{
      const r = await fetch('/chat/session', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(body)
      }});
      const d = await r.json();
      if (r.ok) {{
        sessionId = d.session_id;
        thinking.textContent = d.response;
        thinking.classList.remove('thinking');
        // S10.1 — botones de feedback
        const interactionId = d.session_id + '_t' + d.turn;
        const fbDiv = document.createElement('div');
        fbDiv.className = 'fb-row';
        fbDiv.dataset.id = interactionId;
        fbDiv.innerHTML = `<button class="fb-btn" style="color:#16a34a;border-color:#16a34a" title="Marca esta respuesta como acierto (alimenta el aprendizaje)" onclick="sendFeedback('${{interactionId}}',1,this)">&#10003; Marcar como acierto</button><button class="fb-btn" style="color:#dc2626;border-color:#dc2626" title="Marca esta respuesta como error (alimenta el aprendizaje)" onclick="sendFeedback('${{interactionId}}',-1,this)">&#10007; Marcar como error</button>`;
        thinking.parentElement.appendChild(fbDiv);
      }} else {{
        thinking.textContent = 'Error: ' + (d.detail || r.status);
      }}
    }} catch(e) {{
      thinking.textContent = 'Error de conexión: ' + e.message;
    }}
    btn.disabled = false; input.focus();
  }}

  // S10.1 — enviar feedback (👍 / 👎)
  async function sendFeedback(interactionId, rating, btn) {{
    const row = btn.parentElement;
    try {{
      await fetch('/feedback', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ interaction_id: interactionId, rating }})
      }});
      row.innerHTML = rating === 1
        ? '<span style="font-size:12px;color:#16a34a">&#10003; Guardado como acierto</span>'
        : '<span style="font-size:12px;color:#dc2626">&#10007; Guardado como error</span>';
    }} catch(e) {{
      row.innerHTML = '<span style="font-size:11px;color:#888">Error al enviar feedback</span>';
    }}
  }}

  function clearSession() {{
    if (sessionId) {{
      fetch('/chat/session/' + sessionId, {{method: 'DELETE'}}).catch(()=>{{}});
      sessionId = null;
    }}
    document.getElementById('messages').innerHTML = '';
    addMsg('ai', 'Nueva sesión iniciada. ¿En qué puedo ayudarte?');
  }}

  function handleKey(e) {{
    if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); sendMessage(); }}
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 180) + 'px';
  }}

  // ── Agent ────────────────────────────────────────────────────────
  async function runAgent() {{
    const task = document.getElementById('task-input').value.trim();
    if (!task) return;
    const btn = document.getElementById('agent-btn');
    const maxSteps = parseInt(document.getElementById('max-steps').value) || 12;
    btn.disabled = true;
    btn.textContent = '⏳ Ejecutando…';

    const resultBox = document.getElementById('agent-result');
    const placeholder = document.getElementById('agent-placeholder');
    resultBox.style.display = 'none';
    placeholder.style.display = 'none';
    document.getElementById('agent-result-title').textContent = 'Ejecutando…';
    resultBox.style.display = 'block';
    document.getElementById('agent-answer').textContent = '';
    document.getElementById('agent-steps').innerHTML = '';

    try {{
      const r = await fetch('/agent', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ task, max_steps: maxSteps }})
      }});
      const d = await r.json();
      if (r.ok) {{
        renderAgentResult(d);
      }} else {{
        document.getElementById('agent-answer').textContent = 'Error: ' + (d.detail || r.status);
        document.getElementById('agent-result-title').textContent = 'Error';
      }}
    }} catch(e) {{
      document.getElementById('agent-answer').textContent = 'Error de conexión: ' + e.message;
      document.getElementById('agent-result-title').textContent = 'Error';
    }}
    btn.disabled = false;
    btn.textContent = '▶ Ejecutar agente';
  }}

  function renderAgentResult(d) {{
    const title = document.getElementById('agent-result-title');
    const answerEl = document.getElementById('agent-answer');
    const stepsEl = document.getElementById('agent-steps');

    title.textContent = (d.success ? '✅' : '⚠️') + ` Respuesta (${{d.steps_taken}} pasos · ${{(d.ms/1000).toFixed(1)}}s)`;
    answerEl.textContent = d.answer;

    stepsEl.innerHTML = '';
    d.steps.forEach((step, i) => {{
      const div = document.createElement('div');
      div.className = 'step';
      const badge = step.final_answer !== undefined
        ? '<span class="badge-final">Final</span>'
        : (step.action ? `<span class="badge-action">${{esc(step.action)}}</span>` : '');
      div.innerHTML = `
        <div class="step-header" onclick="toggleStep(this)">
          <span>Paso ${{i+1}} ${{badge}}</span>
          <span>▼</span>
        </div>
        <div class="step-body">
          ${{step.thought ? `<div class="label">Thought</div><div class="value">${{esc(step.thought)}}</div>` : ''}}
          ${{step.action ? `<div class="label">Action</div><div class="value">${{esc(step.action)}}(${{esc(JSON.stringify(step.action_input || {{}}))}})</div>` : ''}}
          ${{step.observation !== undefined ? `<div class="label">Observation</div><div class="obs">${{esc(step.observation)}}</div>` : ''}}
          ${{step.final_answer !== undefined ? `<div class="label">Final Answer</div><div class="value">${{esc(step.final_answer)}}</div>` : ''}}
        </div>`;
      stepsEl.appendChild(div);
    }});
    // Expandir el último paso por defecto
    if (stepsEl.lastChild) {{
      stepsEl.lastChild.querySelector('.step-body').classList.add('open');
    }}
  }}

  function toggleStep(header) {{
    const body = header.nextElementSibling;
    body.classList.toggle('open');
    header.querySelector('span:last-child').textContent =
      body.classList.contains('open') ? '▲' : '▼';
  }}

  // ── Arranque ─────────────────────────────────────────────────────
  fetchHealth();
  setInterval(fetchHealth, 30000);
  addMsg('ai', 'Hola, soy {model_name}. ¿En qué puedo ayudarte?');
  document.getElementById('msg-input').focus();

  // ── Digestor ──────────────────────────────────────────────────────
  let digestorFile = null;

  document.addEventListener('DOMContentLoaded', function() {{
    const drop = document.getElementById('file-drop');
    if (!drop) return;
    drop.addEventListener('dragover', e => {{ e.preventDefault(); drop.classList.add('drag-over'); }});
    drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
    drop.addEventListener('drop', e => {{
      e.preventDefault(); drop.classList.remove('drag-over');
      if (e.dataTransfer.files.length) {{
        digestorFile = e.dataTransfer.files[0];
        document.getElementById('file-name').textContent = '📎 ' + digestorFile.name;
      }}
    }});
  }});

  function fileSelected(e) {{
    if (e.target.files.length) {{
      digestorFile = e.target.files[0];
      document.getElementById('file-name').textContent = '📎 ' + digestorFile.name;
    }}
  }}

  async function runDigestor() {{
    if (!digestorFile) {{ alert('Selecciona un archivo primero'); return; }}
    const task = document.getElementById('digestor-task').value.trim();
    if (!task) {{ alert('Escribe una tarea para el LLM'); return; }}

    const btn = document.getElementById('digestor-btn');
    btn.disabled = true; btn.textContent = '⏳ Procesando...';

    const form = new FormData();
    form.append('file', digestorFile);
    form.append('filename', digestorFile.name);
    form.append('task', task);
    form.append('label_col', document.getElementById('digestor-label-col').value);
    form.append('label_map', document.getElementById('digestor-label-map').value);
    form.append('format', document.getElementById('digestor-format').value);
    form.append('domain', document.getElementById('digestor-domain').value);
    form.append('model_id', document.getElementById('digestor-model-id').value);

    try {{
      const r = await fetch('/digestor/process', {{ method: 'POST', body: form }});
      const d = await r.json();
      if (r.ok) {{
        renderDigestorResult(d);
      }} else {{
        alert('Error: ' + (d.detail || r.status));
      }}
    }} catch(e) {{
      alert('Error de conexión: ' + e.message);
    }}
    btn.disabled = false; btn.textContent = '🔄 Procesar datos';
  }}

  function renderDigestorResult(d) {{
    const box = document.getElementById('digestor-result');
    box.style.display = 'block';
    document.getElementById('digestor-result-title').textContent =
      `📊 Resultado — ${{d.total}} ejemplos`;

    let html = '';
    html += `<div class="metric-row"><span class="metric-label">Semáforo:</span><span class="metric-value"><span class="semaforo-badge semaforo-${{d.semaforo}}">${{d.semaforo}}</span></span></div>`;
    html += `<div class="metric-row"><span class="metric-label">Total ejemplos:</span><span class="metric-value">${{d.total}}</span></div>`;
    if (d.label_counts) {{
      html += `<div class="metric-row"><span class="metric-label">Distribución:</span><span class="metric-value">`;
      Object.entries(d.label_counts).forEach(([k,v]) => {{
        html += `${{k}}: ${{v}} `;
      }});
      html += `</span></div>`;
    }}
    html += `<div class="metric-row"><span class="metric-label">Longitud media:</span><span class="metric-value">${{d.avg_chars}} chars (~${{d.avg_tokens}} tokens)</span></div>`;
    if (d.domain) {{
      html += `<div class="metric-row"><span class="metric-label">Dominio detectado:</span><span class="metric-value">${{d.domain}} (confianza: ${{d.confidence}}%)</span></div>`;
    }}
    document.getElementById('digestor-metrics').innerHTML = html;

    let warningsHtml = '';
    if (d.warnings && d.warnings.length) {{
      d.warnings.forEach(w => {{
        warningsHtml += `<div class="warning-item">⚠️ ${{esc(w)}}</div>`;
      }});
    }}
    document.getElementById('digestor-warnings').innerHTML = warningsHtml;

    if (d.download_url) {{
      const link = document.getElementById('digestor-download');
      link.href = d.download_url;
      link.style.display = 'inline-block';
      link.textContent = '⬇ Descargar dataset (' + d.format + ')';
    }}
  }}

  function esc(s) {{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }}
</script>
</body>
</html>"""
        safe_html = html.encode('utf-8', errors='replace').decode('utf-8')
        return HTMLResponse(content=safe_html)

    # ----------------------------------------------------------------
    # GET /health
    # ----------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health():
        """Estado del servidor: modelo cargado, GPU, VRAM usada, perfil hardware."""
        if _state.is_gguf:
            vram_used = 0.0
        else:
            vram_used = (
                torch.cuda.memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available() else 0.0
            )
        from motor.hardware import detect_hardware
        hw_dict = detect_hardware().to_dict()
        return HealthResponse(
            status        = "ok",
            model         = str(Path(_state.model_path).name),
            base_model    = _state.base_model_id,
            is_adapter    = _state.is_adapter,
            gpu           = _state.gpu_name,
            vram_total_gb = round(_state.vram_total_gb, 2),
            vram_used_gb  = round(vram_used, 2),
            dtype         = _state.dtype_str,
            vision        = _state.vision_ready,
            load_time_s   = _state.load_time_s,
            uptime_s      = round(time.time() - _start_time, 1),
            hardware      = hw_dict,
        )

    # ----------------------------------------------------------------
    # POST /chat  (stateless — sin historial, una petición = una respuesta)
    # ----------------------------------------------------------------

    # ── OpenAI-compatible endpoints (usan modelos del módulo) ────────

    # ----------------------------------------------------------------
    # GET /v1/models  (OpenAI-compatible — Odysseus lo necesita)
    # ----------------------------------------------------------------

    @app.get("/v1/models", response_model=_OAIModelList)
    def list_models_openai():
        """Devuelve la lista de modelos disponibles en formato OpenAI.
        Odysseus llama a este endpoint para descubrir modelos automáticamente."""
        model_name = str(Path(_state.model_path).name)
        return _OAIModelList(
            data=[
                _OAIModelEntry(
                    id       = model_name,
                    owned_by = "motor-de-loras",
                )
            ]
        )

    # ----------------------------------------------------------------
    # POST /v1/chat/completions  (OpenAI-compatible — Odysseus lo usa)
    # ----------------------------------------------------------------

    @app.post("/v1/chat/completions", response_model=_OAIResponse, dependencies=[Depends(_check_api_key)])
    def chat_completions_openai(req: _OAIRequest):
        """
        Chat completions en formato OpenAI-compatible.
        """
        import json as _json
        # max_tokens ausente o <=0 → default generoso (semántica OpenAI:
        # omitido = hasta el límite; capamos en 2048 por latencia)
        max_toks = req.max_tokens if (req.max_tokens or 0) > 0 else 2048
        print(f"[OAI] stream={req.stream} model={req.model} "
              f"msgs={len(req.messages)} max_tok={max_toks} "
              f"temp={req.temperature}")
        if _state.model is None and not _state.is_gguf:
            raise HTTPException(status_code=503, detail="Modelo no cargado.")

        # req.model se ignora (hay un único modelo cargado); avisar si difiere
        # para que el desajuste no pase desapercibido en los logs del servidor
        _loaded_name = str(Path(_state.model_path).name)
        if req.model and req.model != _loaded_name:
            print(f"[OAI] AVISO: cliente pidió model='{req.model}' pero el "
                  f"modelo cargado es '{_loaded_name}' — se usa el cargado.")

        # Si el cliente pide streaming pero no estamos en modo GGUF,
        # advertimos y caemos en modo normal (HuggingFace no soporta SSE)
        if req.stream and not _state.is_gguf:
            print("[OAI] AVISO: streaming solo disponible en modo GGUF, usando no-stream")
            req.stream = False

        # Convertir mensajes al formato interno (normalizando content y
        # preservando los campos de tool-calling, necesarios para que el
        # historial de un loop agéntico llegue íntegro al chat template)
        # Una imagen sin visión activa se perdería al aplanar el content → el
        # modelo respondería sobre la nada. Mejor un 400 explícito que una
        # respuesta silenciosamente equivocada.
        if any(_has_image_part(m.content) for m in req.messages) \
                and not _state.vision_ready:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": ("Este serve está en modo TEXTO: la petición trae "
                                "una imagen y se habría ignorado. Arranca el perfil "
                                "multimodal (MOTOR_MMPROJ con el mmproj del modelo) "
                                "para habilitar visión."),
                    "type": "vision_not_available",
                },
            )

        messages = []
        for m in req.messages:
            d: dict = {"role": m.role, "content": _oai_content_for_model(m.content)}
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.name:
                d["name"] = m.name
            messages.append(d)

        # Inyectar system prompt si el cliente no envió uno
        if not any(m["role"] == "system" for m in messages):
            _DEFAULT_SYSTEM = (
                "Eres un asistente IA local del Motor de LoRAs, ejecutándote en España. "
                "Puedes ayudar con tareas domésticas: organizar archivos, filtrar correos, "
                "buscar documentos, guardar notas, gestionar recordatorios y calendario. "
                "IDIOMA: Responde EXCLUSIVAMENTE en español. "
                "Eres capaz de usar herramientas (function calling) cuando el usuario te las proporcione. "
                "Sé honesto sobre tus limitaciones y no inventes capacidades que no tienes."
            )
            messages.insert(0, {"role": "system", "content": _DEFAULT_SYSTEM})

        # El id OpenAI sirve también como interaction_id del log: el cliente
        # lo recibe (body o chunks SSE) y puede enviarlo a POST /feedback.
        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        # Último mensaje del usuario → clave de agrupación de DPOBuilder.
        # Se aplana SIEMPRE a texto: con visión, content puede ser una lista
        # con un data-URI, que no debe entrar al log (ver _content_for_log).
        user_msg = _content_for_log(next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        ))

        # En modo GGUF, rango seguro: ni muy fría (repeticiones) ni muy caliente
        # (incoherencia). El tope depende de la familia: la ficha oficial de
        # Gemma 4 recomienda temperature=1.0 (capar a 0.7 la degradaba);
        # para Qwen Q4_K_M, >0.7 producía incoherencia (observado 5-jun).
        temp = req.temperature
        if _state.is_gguf:
            _max_temp = 1.0 if _model_family(_state.model_path) == "gemma" else 0.7
            if temp < 0.2:
                print(f"[OAI] AVISO: temperatura {temp} < 0.2, subida a 0.3")
                temp = 0.3
            elif temp > _max_temp:
                print(f"[OAI] AVISO: temperatura {temp} > {_max_temp}, capada")
                temp = _max_temp

        t0 = time.time()

        # ── Tool-calling (solo modo GGUF: llama-cpp lo soporta nativo) ─
        use_tools = bool(req.tools) and _state.is_gguf
        if req.tools and not _state.is_gguf:
            print("[OAI] AVISO: tools solo soportadas en modo GGUF — ignoradas")

        def _run_with_tools():
            """create_chat_completion con tools → (texto, tool_calls, finish_reason, usage)."""
            kw: dict = dict(
                messages       = messages,
                max_tokens     = max_toks,
                temperature    = max(temp, 0.01),
                top_p          = req.top_p,
                repeat_penalty = 1.1,
                tools          = req.tools,
            )
            if req.tool_choice is not None:
                kw["tool_choice"] = req.tool_choice
            with _INFER_LOCK:      # llama-cpp no es thread-safe
                result = _state.llama_model.create_chat_completion(**kw)
            choice = result["choices"][0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls") or None
            text = _strip_gemma_channels(msg.get("content") or "")
            # Fallback Gemma 4: el modelo emite su sintaxis nativa de tool
            # call como texto (llama-cpp no la parsea) → convertir a OpenAI
            if not tool_calls and text:
                parsed = _parse_gemma_tool_calls(text)
                if parsed:
                    tool_calls = parsed
                    text = ""
            finish = choice.get("finish_reason") or "stop"
            if tool_calls:
                finish = "tool_calls"
            return text, tool_calls, finish, result.get("usage") or {}

        # ── Modo streaming (SSE) ──────────────────────────────────────
        if req.stream and _state.is_gguf:
            from fastapi.responses import StreamingResponse
            import json as _json

            # Con tools no hay streaming token a token: los argumentos de
            # una tool call solo sirven completos. Ejecutamos no-stream y
            # emitimos el resultado como un único chunk SSE (compatible
            # con clientes OpenAI).
            if use_tools:
                def _generate_tools():
                    model_name = str(Path(_state.model_path).name)
                    created = int(time.time())
                    try:
                        text, tool_calls, finish, _usage = _run_with_tools()
                    except Exception as e:
                        yield f"data: {_json.dumps({'error': str(e)})}\n\n"
                        return
                    delta: dict = {"role": "assistant"}
                    if text:
                        delta["content"] = text
                    if tool_calls:
                        delta["tool_calls"] = [
                            dict(tc, index=i) for i, tc in enumerate(tool_calls)
                        ]
                    for d, f in ((delta, None), ({}, finish)):
                        yield "data: " + _json.dumps({
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": d, "finish_reason": f}],
                        }) + "\n\n"
                    yield "data: [DONE]\n\n"
                    _log_interaction(
                        chat_id, user_msg, text,
                        ms=int((time.time() - t0) * 1000),
                        endpoint="v1/chat/completions",
                        extra={"finish_reason": finish,
                               **({"tool_calls": tool_calls} if tool_calls else {})},
                    )
                return StreamingResponse(_generate_tools(), media_type="text/event-stream")

            def _generate():
                model_name = str(Path(_state.model_path).name)
                created = int(time.time())
                collected: List[str] = []
                finish_seen = None
                # Retención del canal 'thought' de Gemma: si la respuesta
                # empieza por <|channel>thought, retenemos los chunks hasta
                # su cierre y emitimos solo el contenido posterior.
                hold: List[str] = []
                holding = True

                def _sse(delta, finish):
                    return "data: " + _json.dumps({
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish,
                        }],
                    }) + "\n\n"

                # El lock cubre TODO el generador: cada chunk avanza el modelo,
                # así que soltarlo entre tokens dejaría entrar otra petición y
                # petaría igual. `finally` lo libera aunque el cliente corte
                # (GeneratorExit). Serializa el streaming con el resto: una
                # respuesta en curso bloquea a las siguientes hasta terminar.
                acquired = _INFER_LOCK.acquire()
                try:
                    stream = _state.llama_model.create_chat_completion(
                        messages=messages,
                        max_tokens=max_toks,
                        temperature=max(temp, 0.01),
                        top_p=req.top_p,
                        stream=True,
                    )
                    for chunk in stream:
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = dict(choices[0].get("delta", {}))
                        finish = choices[0].get("finish_reason")
                        if finish:
                            finish_seen = finish
                        piece = delta.get("content")
                        if piece:
                            collected.append(piece)
                        if holding and piece:
                            hold.append(piece)
                            decided, text_out = _gemma_stream_hold("".join(hold))
                            if decided:
                                holding = False
                                delta["content"] = text_out
                            else:
                                delta.pop("content", None)
                                if not delta and not finish:
                                    continue  # nada que emitir aún
                        elif holding and finish:
                            # Terminó mientras reteníamos → soltar lo limpio
                            holding = False
                            buf = _strip_gemma_channels("".join(hold))
                            if buf:
                                delta["content"] = buf
                        yield _sse(delta, finish)
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    err = {"error": str(e)}
                    yield f"data: {_json.dumps(err)}\n\n"
                finally:
                    # Liberar el lock de inferencia SIEMPRE (incl. GeneratorExit
                    # si el cliente corta): si no, el serve quedaría bloqueado
                    # para todos tras un streaming abortado.
                    if acquired:
                        _INFER_LOCK.release()
                    # Se ejecuta también si el cliente desconecta a mitad
                    # (GeneratorExit): logueamos lo acumulado hasta entonces.
                    if collected:
                        _log_interaction(
                            chat_id, user_msg,
                            _strip_gemma_channels("".join(collected)),
                            ms=int((time.time() - t0) * 1000),
                            endpoint="v1/chat/completions",
                            extra=({"finish_reason": finish_seen}
                                   if finish_seen else {"finish_reason": "disconnect"}),
                        )

            return StreamingResponse(_generate(), media_type="text/event-stream")

        # ── Modo normal (no-streaming) ────────────────────────────────
        tool_calls = None
        try:
            if use_tools:
                response_text, tool_calls, finish_reason, usage = _run_with_tools()
            else:
                response_text, usage, finish_reason = _infer(
                    messages, max_toks, temp, req.top_p, return_meta=True
                )
        except Exception as e:
            _raise_inference_error(e)

        model_name = str(Path(_state.model_path).name)
        elapsed_ms = int((time.time() - t0) * 1000)

        _conv_sid, _conv_turn = _oai_conversation_key(messages)
        _log_interaction(
            chat_id, user_msg, response_text,
            ms=elapsed_ms,
            session_id=_conv_sid,
            turn=_conv_turn,
            endpoint="v1/chat/completions",
            extra={"finish_reason": finish_reason,
                   **({"tool_calls": tool_calls} if tool_calls else {})},
        )

        if not usage:
            # Fallback si llama-cpp no devolvió usage: estimación chars//4
            # (_content_for_log aplana: con visión, content puede ser lista)
            prompt_chars = sum(len(_content_for_log(m.get("content")) or "")
                               for m in messages)
            usage = {
                "prompt_tokens":     prompt_chars // 4,
                "completion_tokens": len(response_text) // 4,
                "total_tokens":      (prompt_chars + len(response_text)) // 4,
            }

        return _OAIResponse(
            id      = chat_id,
            created = int(time.time()),
            model   = model_name,
            choices = [
                _OAIChoice(
                    index         = 0,
                    message       = _OAIMessage(
                        role       = "assistant",
                        # OpenAI devuelve content=null cuando es tool call pura
                        content    = response_text if response_text else (None if tool_calls else ""),
                        tool_calls = tool_calls,
                    ),
                    finish_reason = finish_reason,
                )
            ],
            usage   = _OAIUsage(
                prompt_tokens     = usage.get("prompt_tokens", 0),
                completion_tokens = usage.get("completion_tokens", 0),
                total_tokens      = usage.get("total_tokens", 0),
            ),
        )

    # ----------------------------------------------------------------
    # POST /chat  (stateless — sin historial, una petición = una respuesta)
    # ----------------------------------------------------------------

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_check_api_key)])
    def chat(req: ChatRequest):
        """
        Inferencia stateless. Cada petición es independiente.
        Ideal para clasificación, extracción de datos, tareas concretas.

        Ejemplo:
            POST /chat
            {"message": "AAPL is crashing", "system": "Classify sentiment: POSITIVE/NEGATIVE/NEUTRAL"}
        """
        if _state.model is None and not _state.is_gguf:
            raise HTTPException(status_code=503, detail="Modelo no cargado.")

        messages = [
            {"role": "system",    "content": req.system},
            {"role": "user",      "content": req.message},
        ]
        t0 = time.time()
        try:
            response = _infer(messages, req.max_tokens, req.temperature, req.top_p)
        except Exception as e:
            _raise_inference_error(e)

        return ChatResponse(
            response = response,
            model    = str(Path(_state.model_path).name),
            ms       = int((time.time() - t0) * 1000),
        )

    # ----------------------------------------------------------------
    # POST /chat/session  (stateful — con historial por sesión)
    # ----------------------------------------------------------------

    @app.post("/chat/session", response_model=SessionChatResponse, dependencies=[Depends(_check_api_key)])
    def chat_session(req: SessionChatRequest):
        """
        Inferencia con historial de conversación por sesión.
        Ideal para chatbots, asistentes interactivos, modo agente.

        Si no se indica session_id, se crea una nueva sesión automáticamente.
        El session_id devuelto debe incluirse en peticiones posteriores para
        mantener el contexto de la conversación.

        Ejemplo:
            POST /chat/session
            {"message": "¿Cuál es la capital de Francia?"}
            → {"response": "París", "session_id": "abc-123", "turn": 1}

            POST /chat/session
            {"message": "¿Y la de Alemania?", "session_id": "abc-123"}
            → {"response": "Berlín", "session_id": "abc-123", "turn": 2}
        """
        if _state.model is None and not _state.is_gguf:
            raise HTTPException(status_code=503, detail="Modelo no cargado.")

        # Purga de sesiones: TTL de inactividad + tope global
        now = time.time()
        expired = [sid for sid, ts in _state.session_last_seen.items()
                   if now - ts > SESSION_TTL_S]
        for sid in expired:
            _state.sessions.pop(sid, None)
            _state.session_last_seen.pop(sid, None)
        if len(_state.sessions) >= MAX_SESSIONS:
            oldest = sorted(_state.session_last_seen, key=_state.session_last_seen.get)
            for sid in oldest[:len(_state.sessions) - MAX_SESSIONS + 1]:
                _state.sessions.pop(sid, None)
                _state.session_last_seen.pop(sid, None)

        # Crear o recuperar sesión
        session_id = req.session_id or str(uuid.uuid4())
        if session_id not in _state.sessions:
            _state.sessions[session_id] = []
        _state.session_last_seen[session_id] = now

        history = _state.sessions[session_id]
        history.append({"role": "user", "content": req.message})

        # Podar historial si excede el limite (mantener system + ultimos N mensajes)
        if len(history) > MAX_SESSION_MESSAGES:
            # Conservar el system prompt si existe + ultimos mensajes
            system_msg = history[0] if history and history[0]["role"] == "system" else None
            trimmed = history[-(MAX_SESSION_MESSAGES - (1 if system_msg else 0)):]
            if system_msg and trimmed[0]["role"] != "system":
                trimmed = [system_msg] + trimmed
            history.clear()
            history.extend(trimmed)

        messages = [{"role": "system", "content": req.system}] + history
        t0 = time.time()
        try:
            response = _infer(messages, req.max_tokens, req.temperature, req.top_p)
        except Exception as e:
            # Revertir el mensaje añadido si falla
            history.pop()
            _raise_inference_error(e)

        history.append({"role": "assistant", "content": response})

        elapsed_ms = int((time.time() - t0) * 1000)
        turn_n     = len([m for m in history if m["role"] == "user"])
        interaction_id = f"{session_id}_t{turn_n}"

        # S10.1 — guardar interacción en log JSONL
        _log_interaction(
            interaction_id, req.message, response,
            ms=elapsed_ms,
            session_id=session_id,
            turn=turn_n,
            endpoint="chat/session",
        )

        return SessionChatResponse(
            response   = response,
            session_id = session_id,
            turn       = turn_n,
            model      = str(Path(_state.model_path).name),
            ms         = elapsed_ms,
        )

    # ----------------------------------------------------------------
    # DELETE /chat/session  (borrar historial de una sesión)
    # ----------------------------------------------------------------

    @app.delete("/chat/session/{session_id}", dependencies=[Depends(_check_api_key)])
    def delete_session(session_id: str):
        """Elimina el historial de una sesión."""
        if session_id in _state.sessions:
            del _state.sessions[session_id]
            _state.session_last_seen.pop(session_id, None)
            return {"detail": f"Sesión {session_id} eliminada."}
        raise HTTPException(status_code=404, detail=f"Sesión {session_id} no encontrada.")

    # ----------------------------------------------------------------
    # POST /agent  (ReAct loop con herramientas)
    # ----------------------------------------------------------------

    @app.post("/agent", response_model=AgentResponse, dependencies=[Depends(_require_api_key)])
    def agent_run(req: AgentRequest):
        """
        Ejecuta el agente ReAct con la tarea indicada.

        El agente razona paso a paso y usa herramientas (read_file, list_dir,
        shell, http_get) hasta obtener una respuesta final o agotar max_steps.

        Ejemplo:
            POST /agent
            {"task": "¿Cuántos ejemplos tiene el dataset datasets/titanic.jsonl?"}
            → {"answer": "El dataset tiene 891 ejemplos.", "steps": [...], "success": true}
        """
        if _state.model is None and not _state.is_gguf:
            raise HTTPException(status_code=503, detail="Modelo no cargado.")

        from motor.agent import LoRAAgent, DEFAULT_TOOLS
        try:
            from motor.domestic_tools import DOMESTIC_TOOLS
        except Exception:
            DOMESTIC_TOOLS = []

        agent = LoRAAgent(
            infer_fn  = _infer,
            tools     = DEFAULT_TOOLS + DOMESTIC_TOOLS,
            max_steps = req.max_steps,
        )
        t0 = time.time()
        try:
            result = agent.run(req.task)
        except Exception as e:
            _raise_inference_error(e, "Error en el agente")

        d = result.to_dict()
        d["ms"] = int((time.time() - t0) * 1000)
        return AgentResponse(**d)

    # ----------------------------------------------------------------
    # POST /feedback  (S10.1) — thumbs up/down sobre una interacción
    # ----------------------------------------------------------------

    @app.post("/feedback", dependencies=[Depends(_check_api_key)])
    def feedback(req: FeedbackRequest):
        """
        Registra feedback (👍 / 👎) para una interacción previa.

        Busca la entrada con ``interaction_id`` en el log JSONL y actualiza
        su campo ``feedback``. Si no la encuentra, la añade como nueva línea.

        Ejemplo:
            POST /feedback
            {"interaction_id": "abc-123_t2", "rating": 1}
        """
        if not _state.interaction_log_path:
            raise HTTPException(status_code=503, detail="Log de interacciones no configurado.")

        log_path = Path(_state.interaction_log_path)
        if not log_path.exists():
            raise HTTPException(status_code=404, detail="Log de interacciones vacío.")

        # Leer todas las líneas, actualizar la coincidente y reescribir
        updated = False
        lines: List[str] = []
        with open(log_path, encoding="utf-8") as lf:
            for raw in lf:
                raw = raw.rstrip("\n")
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except Exception:
                    lines.append(raw)
                    continue
                if entry.get("id") == req.interaction_id:
                    entry["feedback"] = req.rating
                    if req.comment:
                        entry["feedback_comment"] = req.comment
                    updated = True
                lines.append(json.dumps(entry, ensure_ascii=False))

        if not updated:
            # Añadir entrada mínima de feedback si no existe la interacción
            import datetime as _dt
            lines.append(json.dumps({
                "id":        req.interaction_id,
                "timestamp": _dt.datetime.now(_dt.timezone.utc)
                                .isoformat().replace("+00:00", "Z"),
                "feedback":  req.rating,
                "feedback_comment": req.comment,
            }, ensure_ascii=False))

        with open(log_path, "w", encoding="utf-8") as lf:
            for line in lines:
                lf.write(line + "\n")

        return {"ok": True, "updated": updated, "interaction_id": req.interaction_id}

    # ----------------------------------------------------------------
    # POST /digestor/process — procesar archivo con DataDigestor
    # ----------------------------------------------------------------
    try:
        from fastapi import File, Form
    except ImportError:
        from fastapi import File, Form

    @app.post("/digestor/process")
    async def digestor_process(
        file: bytes = File(...),
        task: str = Form(...),
        filename: str = Form(""),
        label_col: str = Form(""),
        label_map: str = Form(""),
        format: str = Form("chatml"),
        domain: str = Form("auto"),
        model_id: str = Form(""),
    ):
        """
        Recibe un archivo, lo procesa con DataDigestor y devuelve
        metricas + semaforo + link de descarga del dataset generado.
        """
        # Guardar archivo temporal con extension correcta
        import tempfile
        suffix = Path(filename).suffix if filename else ".tmp"
        if not suffix:
            suffix = ".tmp"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(file)
        tmp.close()

        # Construir label_map
        lm = None
        if label_map.strip():
            lm = {}
            for pair in label_map.split(","):
                pair = pair.strip()
                if ":" not in pair:
                    continue
                k, v = pair.split(":", 1)
                k = k.strip(); v = v.strip()
                try: k = int(k)
                except ValueError:
                    try: k = float(k)
                    except ValueError: pass
                lm[k] = v

        # Crear digestor
        from motor.digestor import DataDigestor, detect_file_type
        d = DataDigestor(
            task=task,
            label_col=label_col or None,
            label_map=lm,
            output_format="chatml",
            model_id=model_id or None,
            domain=domain,
            auto_enrich=True,
        )

        # Detectar tipo y cargar
        ft = detect_file_type(tmp.name)
        if ft == "csv":
            d.from_csv(tmp.name)
        elif ft in ("json", "jsonl"):
            d.from_json(tmp.name, text_field="text", label_field=label_col or None)
        elif ft == "txt":
            d.from_txt(tmp.name)
        elif ft == "pdf":
            d.from_pdf(tmp.name)
        elif ft == "docx":
            d.from_docx(tmp.name)
        elif ft == "html":
            d.from_html(tmp.name)
        elif ft == "audio":
            d.from_audio(tmp.name)
        elif ft == "video":
            d.from_video(tmp.name)
        else:
            raise HTTPException(400, f"Formato no soportado: {ft}")

        # Validar
        result = d.validate(verbose=False)

        # Exportar según formato
        import os as _os
        out_dir = Path(tempfile.mkdtemp(prefix="digestor_server_"))
        dl_name = Path(filename or "dataset").stem

        if format == "unsloth":
            out_path = out_dir / f"{dl_name}_unsloth.jsonl"
            d.to_unsloth(str(out_path))
        elif format == "llamafactory":
            d.to_llamafactory(str(out_dir), dl_name)
            out_path = out_dir / f"{dl_name}.json"
        elif format == "axolotl":
            d.to_axolotl(str(out_dir), dl_name)
            out_path = out_dir / f"{dl_name}.jsonl"
        else:
            out_path = out_dir / f"{dl_name}.jsonl"
            d.to_jsonl(str(out_path))

        # Servir archivo para descarga (lo mantenemos accesible)
        from fastapi.responses import FileResponse
        dl_url = f"/digestor/download/{out_path.name}"
        # Guardamos referencia global para el endpoint de descarga
        if not hasattr(app.state, "digestor_files"):
            app.state.digestor_files = {}
        app.state.digestor_files[out_path.name] = str(out_path)

        return {
            "total": result.get("total", 0),
            "semaforo": result.get("semaforo", "ROJO"),
            "warnings": result.get("warnings", []),
            "label_counts": result.get("label_counts", {}),
            "avg_chars": round(result.get("avg_chars", 0)),
            "avg_tokens": round(result.get("avg_tokens", 0)),
            "domain": getattr(d, "_detected_domain", None),
            "confidence": getattr(d, "_domain_confidence", 0),
            "format": format,
            "download_url": dl_url,
        }

    # ----------------------------------------------------------------
    # GET /digestor/download/{name} — descargar dataset generado
    # ----------------------------------------------------------------
    @app.get("/digestor/download/{name}")
    async def digestor_download(name: str):
        from fastapi.responses import FileResponse
        if not hasattr(app.state, "digestor_files") or name not in app.state.digestor_files:
            raise HTTPException(404, "Archivo no encontrado")
        return FileResponse(app.state.digestor_files[name], filename=name)

    return app


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Watchdog — recarga el modelo cuando el worker publica un candidato
# ---------------------------------------------------------------------------

_KNOWN_FAMILIES = ("gemma", "qwen", "llama", "mistral", "phi", "deepseek", "smollm")


def _model_family(name: str) -> Optional[str]:
    """
    Familia de un modelo a partir de su nombre, ruta o id HuggingFace.
    'modelos/gemma-4-12B-it-Q4_K_M.gguf' → 'gemma';
    'Qwen/Qwen2.5-7B-Instruct' → 'qwen'. None si no se reconoce.
    """
    n = str(name).lower()
    for fam in _KNOWN_FAMILIES:
        if fam in n:
            return fam
    return None


def _check_promotion(flag: Path, candidate: Path, current: Path) -> Optional[str]:
    """
    Una pasada del watchdog (extraída para poder testearla).

    Guard de linaje (auditoría 10-jun-2026): si el flag declara un
    `base_model` de familia distinta a la del modelo servido, la promoción
    se RECHAZA (flag → rejected.flag) en vez de sustituir el modelo en
    silencio. Evita que un ciclo CL configurado con Qwen reemplace a Gemma.

    El flag puede declarar `gguf` con la ruta del candidato (formato del
    odysseus_bridge); si no, se usa el `candidate` clásico.

    Devuelve "promoted" | "rejected" | None (sin flag).
    """
    if not flag.exists():
        return None

    meta: dict = {}
    try:
        parsed = json.loads(flag.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            meta = parsed
    except Exception:
        pass  # flag antiguo / texto plano → sin metadatos

    # ── Guard de linaje ───────────────────────────────────────────────
    flag_base  = meta.get("base_model")
    served_fam = _model_family(_state.model_path)
    if flag_base and served_fam:
        if _model_family(flag_base) != served_fam:
            rejected = flag.with_name("rejected.flag")
            shutil.move(str(flag), str(rejected))
            print(f"[Watchdog] ❌ Promoción RECHAZADA: el candidato deriva de "
                  f"'{flag_base}' pero el modelo servido es de familia "
                  f"'{served_fam}'. Flag movido a {rejected.name}.")
            return "rejected"
    elif not flag_base:
        print("[Watchdog] ⚠️ ready.flag sin 'base_model': no se puede verificar "
              "el linaje del candidato (formato antiguo). Promoviendo igualmente.")

    src = Path(meta["gguf"]) if meta.get("gguf") else candidate
    if src.exists():
        # Reemplaza atómicamente en Linux (mismo filesystem)
        shutil.move(str(src), str(current))
        print(f"[Watchdog] {src.name} promovido a {current.name}")
    if current.exists():
        load_model(str(current))
        print("[Watchdog] Modelo recargado correctamente.")
    flag.unlink(missing_ok=True)
    return "promoted"


def _watchdog(promotion_dir: Path, interval: int = 30) -> None:
    """
    Hilo daemon que monitoriza promotion/ready.flag cada `interval` segundos.
    Cuando lo detecta:
      1. Verifica el linaje del candidato (rechaza familias distintas)
      2. Mueve el GGUF candidato → modelos/current.gguf (atómico)
      3. Recarga el modelo en _state
      4. Borra el flag
    """
    flag      = promotion_dir / "ready.flag"
    candidate = Path("modelos") / "candidate.gguf"
    current   = Path("modelos") / "current.gguf"

    while True:
        try:
            _check_promotion(flag, candidate, current)
        except Exception as exc:  # noqa: BLE001
            print(f"[Watchdog] Error al promover candidato: {exc}")
            flag.unlink(missing_ok=True)
        time.sleep(interval)


# Punto de entrada: iniciar servidor
# ---------------------------------------------------------------------------

def run_server(
    model_path:  str,
    host:        str  = "127.0.0.1",
    port:        int  = 8000,
    base_model:  Optional[str] = None,
    cache_dir:   Optional[str] = None,
    api_key:     Optional[str] = None,
    reload:      bool = False,
    ui_only:     bool = False,
) -> None:
    """
    Carga el modelo y arranca el servidor uvicorn.
    Bloqueante — corre hasta Ctrl+C.

    Si ui_only=True, arranca sin modelo (solo pestaña Digestor).
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "[Server] uvicorn no instalado.\n"
            "Ejecuta: pip install fastapi uvicorn"
        )

    if ui_only:
        _state.model_path = "(UI only)"
        _state.gpu_name = "N/A"
        print("[Server] Modo UI only — sin modelo cargado. Solo Digestor disponible.")
    else:
        load_model(model_path, base_model=base_model, cache_dir=cache_dir, api_key=api_key)

    app = create_app()

    # Arrancar watchdog de promoción en hilo daemon
    promotion_dir = Path("promotion")
    promotion_dir.mkdir(exist_ok=True)
    t = threading.Thread(
        target=_watchdog, args=(promotion_dir,), daemon=True, name="watchdog"
    )
    t.start()
    print("[Server] Watchdog activo — monitorizando promotion/ cada 30s")

    print(f"\n[Server] Servidor arrancando en http://{host}:{port}")
    if not ui_only:
        print(f"[Server] Documentación interactiva: http://{host}:{port}/docs")
    if api_key:
        print(f"[Server] Autenticación: Bearer token requerido")
    else:
        print(f"[Server] Sin autenticación")
        if host not in ("127.0.0.1", "localhost", "::1"):
            print(f"[Server] ⚠️ ADVERTENCIA DE SEGURIDAD: escuchando en {host} "
                  f"(accesible desde la red) SIN API key.\n"
                  f"[Server]   Cualquier dispositivo de la red puede usar el modelo.\n"
                  f"[Server]   POST /agent queda deshabilitado (403) hasta definir --api-key.")
    print(f"[Server] Pulsa Ctrl+C para detener\n")

    uvicorn.run(app, host=host, port=port, reload=reload)
