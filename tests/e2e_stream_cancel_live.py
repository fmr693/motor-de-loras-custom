"""
e2e_stream_cancel_live.py — cancelación de stream contra un servidor REAL
=========================================================================
Fuera de la suite pytest a propósito: levanta uvicorn en un puerto y corta la
conexión de verdad. `TestClient` NO sirve para esto — bajo él
`request.is_disconnected()` nunca devuelve True (verificado), así que un test
con TestClient pasaría en verde sin probar nada. Esta es la lección de la
ronda 1 de estrés: un mock que no reproduce la realidad es teatro.

Uso:
    python tests/e2e_stream_cancel_live.py

No necesita GPU ni el modelo real: usa un modelo falso con latencia por token.
Lo que se mide es el comportamiento del SERVIDOR, no del modelo.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import uvicorn

from motor import server
from motor.server import create_app


TOTAL_CHUNKS = 300
LATENCIA_TOKEN = 0.01     # 10 ms/token ~ velocidad real de un 12B en GPU
PUERTO = 8899
CORTE_TRAS = 5            # chunks que lee el cliente antes de irse

# Escenario tools: el camino con herramientas es no-stream POR DENTRO (los
# argumentos de una tool call solo sirven completos), así que el modelo tarda
# todo esto ANTES de emitir el primer chunk. El cliente se va a mitad.
LATENCIA_TOOLS = 2.0
CORTE_TOOLS = 0.5


class LlamaLento:
    """Modelo falso que cuenta los chunks realmente producidos."""

    def __init__(self):
        self.producidos = 0
        self.inferencias_tools = 0

    def create_chat_completion(self, messages, stream=False, **kw):
        if kw.get("tools"):
            # Inferencia lenta y COMPLETA antes del primer chunk.
            self.inferencias_tools += 1
            time.sleep(LATENCIA_TOOLS)
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": None,
                                "tool_calls": [{
                                    "id": "call_1", "type": "function",
                                    "function": {"name": "file_organize",
                                                 "arguments": '{"path": "~/Descargas"}'},
                                }]},
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            }
        if not stream:
            return {
                "choices": [{"message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        def gen():
            yield {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
            for i in range(TOTAL_CHUNKS):
                time.sleep(LATENCIA_TOKEN)
                self.producidos += 1
                yield {"choices": [{"delta": {"content": f"t{i} "}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return gen()


def main() -> int:
    modelo = LlamaLento()
    st = server._state
    st.model = None
    st.llama_model = modelo
    st.is_gguf = True
    st.model_path = "modelos/gemma-test.gguf"
    st.api_key = None
    st.interaction_log_path = str(Path(__file__).parent / "_e2e_stream_log.jsonl")
    st.sessions = {}

    config = uvicorn.Config(create_app(), host="127.0.0.1", port=PUERTO, log_level="error")
    servidor = uvicorn.Server(config)
    hilo = threading.Thread(target=servidor.run, daemon=True)
    hilo.start()

    # esperar a que escuche
    for _ in range(100):
        if servidor.started:
            break
        time.sleep(0.05)
    else:
        print("[FALLO] el servidor no arrancó")
        return 1

    print(f"Servidor en :{PUERTO} | generación de {TOTAL_CHUNKS} chunks "
          f"a {LATENCIA_TOKEN*1000:.0f} ms/token (~{TOTAL_CHUNKS*LATENCIA_TOKEN:.0f} s si drena)")

    payload = {
        "model": "gemma",
        "messages": [{"role": "user", "content": "escribe algo muy largo"}],
        "stream": True,
        "max_tokens": TOTAL_CHUNKS,
    }

    t0 = time.time()
    leidos = 0
    with httpx.Client(timeout=30) as c:
        with c.stream("POST", f"http://127.0.0.1:{PUERTO}/v1/chat/completions",
                      json=payload) as r:
            assert r.status_code == 200, f"status {r.status_code}"
            for _ in r.iter_lines():
                leidos += 1
                if leidos >= CORTE_TRAS:
                    break      # el cliente ABANDONA: se cierra la conexión
    t_corte = time.time() - t0
    print(f"Cliente cortó tras {leidos} chunks ({t_corte:.2f} s)")

    # Dar margen a que el servidor reaccione; si no cancelara, seguiría
    # generando hasta TOTAL_CHUNKS.
    margen = 2.0
    time.sleep(margen)
    producidos = modelo.producidos

    print(f"Chunks producidos por el modelo: {producidos} / {TOTAL_CHUNKS}")

    ok = producidos < TOTAL_CHUNKS * 0.5
    if ok:
        print(f"[OK] La generación se detuvo al desconectar el cliente "
              f"({producidos} chunks, no {TOTAL_CHUNKS}).")
    else:
        print(f"[FALLO] El servidor siguió generando ({producidos}/{TOTAL_CHUNKS}): "
              f"el stream abandonado NO se cancela.")

    # --- LO QUE DE VERDAD IMPORTA -------------------------------------------
    # `_INFER_LOCK` es un RLock: solo lo libera el thread que lo tomó. Si el
    # generador abandonado no llega a su `finally` en ese thread, el lock queda
    # TOMADO y el serve deja de atender a todo el mundo. Un stream cortado no
    # puede tumbar el servicio.
    print("\nComprobando que el serve sigue vivo tras el corte...")
    sirve = False
    t1 = time.time()
    try:
        with httpx.Client(timeout=10) as c:
            r2 = c.post(f"http://127.0.0.1:{PUERTO}/v1/chat/completions", json={
                "model": "gemma",
                "messages": [{"role": "user", "content": "sigues vivo?"}],
                "stream": False,
                "max_tokens": 5,
            })
        sirve = r2.status_code == 200
        print(f"  respuesta en {time.time() - t1:.2f} s -> HTTP {r2.status_code}")
    except Exception as e:
        print(f"  [FALLO] el serve NO respondió ({type(e).__name__}): "
              f"_INFER_LOCK quedó tomado por el stream abandonado")

    if sirve:
        print("[OK] El serve sigue atendiendo: el lock se liberó.")
    else:
        ok = False
        print("[FALLO] Serve bloqueado tras un stream abandonado (lock huérfano).")

    # --- REGLA 19: la serialización NO puede haberse roto --------------------
    # `_INFER_LOCK` pasó de RLock a Semaphore para poder liberarlo desde otro
    # thread. Eso no debe costar la exclusión mutua: dos llamadas simultáneas
    # dentro de llama-cpp = segfault (medido en la ronda 2 de estrés).
    print("\nComprobando que la serialización sigue en pie (Regla 19)...")
    solapes = {"max": 0}
    dentro = {"n": 0}
    cerrojo = threading.Lock()
    original = modelo.create_chat_completion

    def espia(*a, **kw):
        with cerrojo:
            dentro["n"] += 1
            solapes["max"] = max(solapes["max"], dentro["n"])
        try:
            time.sleep(0.05)          # tiempo dentro del "modelo"
            return original(*a, **kw)
        finally:
            with cerrojo:
                dentro["n"] -= 1

    modelo.create_chat_completion = espia
    errores = []

    def pedir():
        try:
            with httpx.Client(timeout=20) as c:
                r = c.post(f"http://127.0.0.1:{PUERTO}/v1/chat/completions", json={
                    "model": "gemma",
                    "messages": [{"role": "user", "content": "hola"}],
                    "stream": False, "max_tokens": 5,
                })
                if r.status_code != 200:
                    errores.append(r.status_code)
        except Exception as e:
            errores.append(type(e).__name__)

    hilos = [threading.Thread(target=pedir) for _ in range(8)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout=25)

    print(f"  8 peticiones concurrentes | errores: {len(errores)} | "
          f"máximo simultáneo dentro del modelo: {solapes['max']}")
    if solapes["max"] <= 1 and not errores:
        print("[OK] Sigue serializando: nunca dos peticiones dentro del modelo.")
    else:
        ok = False
        print(f"[FALLO] Serialización rota (solape={solapes['max']}, errores={errores}): "
              "con llama-cpp real esto sería un segfault.")

    # --- EL DATO NO SE TIRA: tools + cliente que abandona --------------------
    # El camino con herramientas es no-stream por dentro: cuando se emite el
    # primer chunk, el modelo YA hizo todo el trabajo. Si el cliente se fue
    # mientras generaba, Starlette abandona el generador en ese primer yield y
    # todo lo que haya DESPUÉS del último yield no corre jamás (ni un `finally`:
    # dependería de que el GC cerrase el generador). Ahí se perdía una
    # interacción ya generada — y las de herramientas son las que más valen
    # para el activo. Regla 11: el log es el activo.
    print("\nComprobando que una interacción con tools no se pierde al abandonar...")
    log_path = Path(st.interaction_log_path)

    def _lineas_log():
        if not log_path.exists():
            return 0
        return len([l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()])

    antes = _lineas_log()
    inferencias_antes = modelo.inferencias_tools
    payload_tools = {
        "model": "gemma",
        "messages": [{"role": "user", "content": "organiza mis descargas"}],
        "stream": True,
        "tools": [{"type": "function",
                   "function": {"name": "file_organize", "description": "organiza",
                                "parameters": {"type": "object"}}}],
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(30, read=CORTE_TOOLS)) as c:
            with c.stream("POST", f"http://127.0.0.1:{PUERTO}/v1/chat/completions",
                          json=payload_tools) as r:
                for _ in r.iter_lines():
                    pass
    except httpx.ReadTimeout:
        pass          # el cliente ABANDONA mientras el modelo genera

    time.sleep(LATENCIA_TOOLS + 1.5)     # dejar terminar la inferencia en curso
    inferencias = modelo.inferencias_tools - inferencias_antes
    nuevas = _lineas_log() - antes
    print(f"  cliente cortó a los {CORTE_TOOLS}s | inferencias del modelo: "
          f"{inferencias} | interacciones nuevas en el log: {nuevas}")
    if inferencias >= 1 and nuevas >= 1:
        print("[OK] La interacción abandonada quedó en el log: el dato no se tira.")
    else:
        ok = False
        print(f"[FALLO] El modelo generó {inferencias} respuesta(s) y el log ganó "
              f"{nuevas}: se PIERDE dato de entrenamiento ya producido.")

    servidor.should_exit = True
    hilo.join(timeout=5)

    if log_path.exists():
        print(f"Log total del arnés: {_lineas_log()} línea(s)")
        log_path.unlink()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
