"""
test_stream_cancel.py
=====================
Cancelación de streaming abandonado (última deuda de resiliencia).

Problema (ronda 3 de estrés): un cliente que corta un stream a mitad NO
cancelaba la generación en servidor. El `finally` liberaba `_INFER_LOCK`, sí,
pero el bucle `for chunk in stream:` seguía pidiendo tokens al modelo hasta
agotar `max_tokens` — el serve "drenaba" la respuesta entera para nadie,
reteniendo el lock y encolando a los demás clientes. Con Odysseus y Hermes
sobre el mismo serve, cerrar una pestaña bloqueaba al siguiente usuario
durante toda la generación restante.

Fix: `_stream_vigilando_desconexion` envuelve el generador síncrono (que sigue
corriendo en el threadpool, Regla 21) y comprueba la desconexión entre chunks;
al detectarla marca un `threading.Event` que el bucle de generación consulta
para salir por su `finally`.

AVISO SOBRE EL ALCANCE DE ESTOS TESTS
--------------------------------------
`TestClient` **no simula la desconexión del cliente**: `is_disconnected()`
nunca devuelve True bajo él (verificado). Por eso aquí se prueba el helper en
aislamiento, con un request falso, que es lo que sí se puede afirmar en CI.
La verificación contra un servidor real (uvicorn + cliente que corta de
verdad) está en `tests/e2e_stream_cancel_live.py`, fuera de la suite.
"""

import asyncio
import threading

import pytest

from motor.server import _stream_vigilando_desconexion


TOTAL = 200          # "max_tokens" del modelo falso
CORTE_EN = 5         # el cliente se va tras este chunk


class RequestFalso:
    """Request mínimo: se declara desconectado a partir del chunk N."""

    def __init__(self, desconecta_en=None):
        self.desconecta_en = desconecta_en
        self.consultas = 0

    async def is_disconnected(self):
        self.consultas += 1
        if self.desconecta_en is None:
            return False
        return self.consultas > self.desconecta_en


def _generador_contador(cancelado: threading.Event, contador: dict):
    """
    Imita el bucle de streaming del servidor: produce hasta TOTAL chunks y
    consulta `cancelado` en cada vuelta, como hace `_generate()` real.
    """
    def gen():
        try:
            for i in range(TOTAL):
                if cancelado.is_set():
                    contador["cancelado_en"] = i
                    break
                contador["producidos"] += 1
                yield f"data: chunk {i}\n\n"
        finally:
            contador["finally_ejecutado"] = True
    return gen()


def _consumir(request, consumir_todo=True):
    """Ejecuta el envoltorio async y devuelve (chunks_recibidos, contador)."""
    cancelado = threading.Event()
    contador = {"producidos": 0, "finally_ejecutado": False, "cancelado_en": None}
    gen = _generador_contador(cancelado, contador)

    async def run():
        recibidos = 0
        async for _ in _stream_vigilando_desconexion(gen, request, cancelado):
            recibidos += 1
            if not consumir_todo and recibidos >= CORTE_EN:
                break
        return recibidos

    recibidos = asyncio.run(run())
    return recibidos, contador, cancelado


# --- el contrato ---------------------------------------------------------------

def test_desconexion_detiene_la_generacion():
    """Si el cliente se va, el modelo deja de producir muy por debajo del total."""
    request = RequestFalso(desconecta_en=CORTE_EN)
    _, contador, _ = _consumir(request)

    assert contador["producidos"] < TOTAL, (
        f"se drenó la generación entera ({contador['producidos']}/{TOTAL}) "
        "pese a la desconexión"
    )
    assert contador["producidos"] <= CORTE_EN + 2, (
        f"el corte llegó tarde: {contador['producidos']} chunks tras desconectar "
        f"en el {CORTE_EN}"
    )


def test_desconexion_marca_el_evento_de_cancelacion():
    """Es el Event lo que el bucle de generación real consulta para parar."""
    request = RequestFalso(desconecta_en=CORTE_EN)
    _, _, cancelado = _consumir(request)

    assert cancelado.is_set()


def test_el_finally_del_generador_se_ejecuta_al_cortar():
    """
    Crítico: el `finally` del generador libera `_INFER_LOCK` y loguea lo
    producido. Debe correr SIEMPRE y de forma determinista al cortar — por eso
    el envoltorio cierra el generador explícitamente en vez de dejarlo al GC
    (que no garantiza ni cuándo ni en qué thread).
    """
    request = RequestFalso(desconecta_en=CORTE_EN)
    _, contador, _ = _consumir(request)

    assert contador["finally_ejecutado"], (
        "el generador quedó suspendido sin ejecutar su finally: el lock no se "
        "liberaría de forma determinista"
    )


def test_cliente_que_consume_todo_no_se_corta():
    """Regresión: sin desconexión, la generación llega al final."""
    request = RequestFalso(desconecta_en=None)
    recibidos, contador, _ = _consumir(request)

    assert contador["producidos"] == TOTAL
    assert recibidos == TOTAL
    assert contador["cancelado_en"] is None


def test_abandono_del_consumidor_tambien_cancela():
    """
    Aunque `is_disconnected()` no llegue a dispararse, si quien consume el
    stream se va (rompe el bucle), el `finally` del envoltorio debe marcar la
    cancelación igualmente — es la red de seguridad.
    """
    request = RequestFalso(desconecta_en=None)
    _, contador, cancelado = _consumir(request, consumir_todo=False)

    assert cancelado.is_set(), "el finally del envoltorio debe cancelar al abandonar"
    assert contador["producidos"] < TOTAL


def test_no_consulta_la_desconexion_de_mas():
    """Una comprobación por chunk: barata, sin polling extra."""
    request = RequestFalso(desconecta_en=None)
    recibidos, _, _ = _consumir(request)

    assert request.consultas == recibidos


# --- el lock de inferencia debe poder liberarse desde OTRO thread --------------

def test_infer_lock_se_libera_desde_otro_thread():
    """
    Núcleo del arreglo. `_INFER_LOCK` era un RLock: solo lo liberaba el thread
    que lo tomó. En streaming el generador lo adquiere en un worker y, si el
    cliente abandona, quien lo cierra es el GC DESDE OTRO THREAD → el release
    lanzaba "cannot release un-acquired lock" y el lock quedaba TOMADO PARA
    SIEMPRE. El serve parecía sobrevivir solo porque RLock es reentrante y la
    siguiente petición podía caer en el mismo worker; en otro, se colgaba.

    Con Semaphore no hay dueño: cualquier thread puede liberarlo.
    """
    from motor.server import _INFER_LOCK

    assert _INFER_LOCK.acquire(timeout=1), "no se pudo tomar el lock"

    fallo = []

    def liberar_desde_otro_thread():
        try:
            _INFER_LOCK.release()
        except RuntimeError as e:      # lo que hacía el RLock
            fallo.append(str(e))

    t = threading.Thread(target=liberar_desde_otro_thread)
    t.start()
    t.join(timeout=5)

    assert not fallo, f"el lock no se puede liberar desde otro thread: {fallo}"
    # y queda realmente libre para el siguiente
    assert _INFER_LOCK.acquire(timeout=1), "el lock quedó tomado tras liberarlo"
    _INFER_LOCK.release()


def test_infer_lock_sigue_siendo_exclusivo():
    """
    Regresión de la Regla 19: el cambio de primitivo NO puede costar la
    exclusión mutua — dos llamadas a la vez dentro de llama-cpp = segfault.
    """
    from motor.server import _INFER_LOCK

    assert _INFER_LOCK.acquire(timeout=1)
    try:
        # un segundo acquire NO debe entrar mientras el primero lo tiene
        assert not _INFER_LOCK.acquire(blocking=False), (
            "el lock dejó entrar a dos a la vez: se perdió la serialización"
        )
    finally:
        _INFER_LOCK.release()
