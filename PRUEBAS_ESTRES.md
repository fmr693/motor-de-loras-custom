# Bitácora de pruebas de estrés — el proceso de aprendizaje

> Este documento no es un informe de resultados: es la **bitácora del aprendizaje**
> que emerge al someter el proyecto a estrés real contra hardware (RTX 4080), en vez
> de solo a la suite de tests. Registra el patrón que se repite, la anatomía de cada
> hallazgo y los principios que van destilándose. Complementa a `SESION.md` (hitos) y
> al README (qué es y cómo se usa).

---

## La tesis (lo que estas pruebas nos están enseñando)

> **Una suite de tests en verde mide que el código hace lo que el test dice, no que el
> sistema resista la realidad. Un test puede ser teatro: si mockea una cadena o prueba
> un único caso feliz, pasa en verde mientras el sistema real se rompe.**

Las tres rondas confirman lo mismo: **cada vez que apretamos de verdad (E2E, contra el
serve real, con entradas hostiles), aparecen costuras que 600+ tests en verde no veían.**
No porque los tests estén mal, sino porque prueban lo que *imaginamos* que puede fallar.
El estrés prueba lo que *de verdad* falla.

Dos ejemplos crudos de "test-teatro" detectados en estas rondas:

1. **El fix de desbordamiento de contexto (15-jul)** tenía un test que *mockeaba el
   mensaje de error de llama-cpp*. El mensaje real cambió entre versiones/caminos de
   código, el fix quedó muerto para el caso de visión — y el test seguía en verde,
   porque comprobaba el mock, no el sistema.
2. **El multimodal recién "verificado E2E"** (un círculo rojo con un 42) tenía un
   **crash de concurrencia latente**: 2 peticiones simultáneas tumbaban el proceso.
   Ningún test unitario lo iba a pillar; una sola imagen "correcta, sin alucinar" tampoco.

---

## Anatomía de un hallazgo (el método)

Cada costura sigue el mismo ciclo, y documentarlo es parte del valor:

```
1. HIPÓTESIS HOSTIL   → "¿qué pasa si mando 2 peticiones a la vez / una imagen rota /
                          un JSON corrupto / corto el streaming a la mitad?"
2. MEDICIÓN E2E       → contra el serve real, observando el comportamiento (no el test)
3. COSTURA            → 500 opaco / crash / dato envenenado / respuesta silenciosamente
                          equivocada / cuelgue
4. CAUSA RAÍZ         → por qué (acoplamiento a un mensaje, thread-safety, validación
                          ausente, degradación deshonesta...)
5. FIX + PRINCIPIO    → arreglo mínimo + la regla que evita la familia entera del fallo
6. TEST QUE NO MIENTA → parametrizado con lo observado en vivo, no con lo que imaginamos
7. RE-VERIFICACIÓN    → repetir el ataque contra el sistema ya arreglado
```

El paso 6 es el que rompe el ciclo del test-teatro: **el test se escribe DESPUÉS de ver
el fallo real, con el dato real**, no antes con una suposición.

---

## Taxonomía de las costuras (los "tipos" que reaparecen)

Los fallos no son aleatorios; caen en familias. Reconocerlas permite anticiparlas:

| Familia | Síntoma | Ejemplos vistos | Principio que la ataja |
|---|---|---|---|
| **Acoplamiento a texto de terceros** | un fix depende de una cadena de error que cambia | overflow 500 (msg de llama-cpp cambió) | reconocer por lista de marcadores, test con los mensajes reales |
| **Thread-safety** | crash bajo concurrencia | 2 peticiones → segfault | serializar el recurso no reentrante |
| **Degradación deshonesta** | anuncia una capacidad que no cumple | `/health vision:true` con mmproj roto | validar antes de anunciar; degradar con aviso |
| **Ignorar en silencio** | descarta entrada sin avisar | imagen aplanada a texto; PDF→0 ejemplos | 400 explícito o aviso claro; nunca salida muda equivocada |
| **Envenenar el dato** | mete basura en el log/dataset | base64 en interaction_log; label lista | higiene en la frontera; el dato de entrenamiento es sagrado |
| **Frágil ante lo malformado** | una entrada mala tumba todo el lote | 1 línea JSON rota aborta el manifiesto | resiliencia por elemento (saltar con reporte) |
| **500 por culpa del cliente** | error del usuario disfrazado de fallo del servidor | imagen inválida, contexto desbordado | traducir a 4xx accionable |

---

## Rondas

### Ronda 1 — Multimodal recién nacido (2 fallos)

Contexto: acabábamos de cablear la visión (mmproj). "Verificado" con una imagen.

| # | Costura | Causa raíz | Familia | Fix |
|---|---|---|---|---|
| 1 | overflow con visión → 500 | el fix del 15-jul solo conocía `"exceed context window"`; el chat handler dice `"Prompt exceeds n_ctx"` | acoplamiento a texto | `_CTX_OVERFLOW_MARKERS` (lista) + test parametrizado con ambos |
| 2 | mmproj corrupto → `/health vision:true`, luego 500 por imagen | llama-cpp carga el proyector de forma **diferida**; construir el handler no valida nada | degradación deshonesta | validar cabecera GGUF antes de declarar visión |

**Lección de la ronda 1:** *el test del fix de overflow mockeaba la cadena antigua → pasaba
en verde mientras el caso real fallaba.* El mock era teatro.

### Ronda 2 — Contra las cuerdas + concurrencia (4 fallos, 1 crítico)

Contexto: batería hostil (imágenes malformadas, multi-imagen, trampas cognitivas,
concurrencia, Digestor adversario).

| # | Costura | Causa raíz | Familia | Fix |
|---|---|---|---|---|
| 1 🔴 | **2 peticiones concurrentes tumban el serve** | llama-cpp no es thread-safe; FastAPI despacha `def` en threadpool | thread-safety | `_INFER_LOCK` (RLock) serializa las 4 llamadas al modelo |
| 2 | imagen malformada → 500 opaco | error del cliente sin traducir | 500 por culpa del cliente | `_IMAGE_ERROR_MARKERS` → 400 `invalid_image` |
| 3 | 1 línea JSON corrupta aborta el manifiesto entero | `json.loads` sin protección por línea | frágil ante lo malformado | lectura resiliente por línea (saltar con aviso) |
| 4 | label no escalar (lista/dict) → `TypeError: unhashable` | lookup en `label_map` sin guardar el tipo | envenenar/crashear | descartar como "sin etiqueta" |

**Lección de la ronda 2:** el fallo crítico era **preexistente** y afectaba al driver
diario (Odysseus + Hermes sobre el mismo serve). No lo introdujo el multimodal; el
multimodal solo dio la excusa para estresar concurrencia por primera vez. *Lo que nunca
habíamos apretado, nunca habíamos visto romperse.*

Lo que **sí aguantó** en la ronda 2 (importante documentarlo — la resiliencia también se
mide por lo que resiste): F4 contra manifiestos EXIST reales (idéntico al script bespoke),
visión real anti-alucinación (no inventa sobre imagen en blanco, no cede ante premisa
falsa, cuenta 11 correctamente), multi-imagen, visión+tools, visión+streaming.

### Ronda 3 — Fuzzing, streaming abandonado, seguridad y Digestor (3 fallos + 1 límite de diseño)

Contexto: la más exhaustiva. Cinco frentes: fuzzing de payloads HTTP, corte de
streaming bajo carga, seguridad/sistema, presión de recursos, Digestor profundo.

**Lo que aguantó** (resiliencia confirmada — cada vez cubre más superficie):
- **Fuzzing HTTP** (18 payloads): JSON roto, body vacío/null/lista, tipos erróneos,
  campos ausentes → Pydantic responde **422/400** limpiamente; el modelo tolera
  content null/número/dict con 200. Solo 1 de 18 dio 500 (ver hallazgo 1).
- **Seguridad**: `/agent` sin API key → **403**; endpoint inexistente → 404; método
  erróneo → 405.
- **Carga sostenida**: 40 peticiones secuenciales → 40/40 OK, **sin degradación**
  (2.80 s primeras vs 2.99 s últimas — plano, sin fuga de memoria/sesiones).
- **Corte de streaming**: cortar a mitad **no produce deadlock** — el `_INFER_LOCK`
  se libera en `finally` (incl. `GeneratorExit`). El serve drena y se recupera.
- **Digestor**: distill con 10k turnos, turno de 640 KB, bytes de control, vacío →
  todo 0.0-0.1 s sin crash. Knowledge sobre doc de 2 MB → 1809 ejemplos en 0.1 s
  (sin O(n²)). CSV multilínea entre comillas, JSON de 20k, JSONL de 50k → OK.

| # | Costura | Causa raíz | Familia | Fix |
|---|---|---|---|---|
| 1 | surrogate suelto en content → 500 | texto malformado sin traducir | 500 por culpa del cliente | `_ENCODING_ERROR_MARKERS` → 400 `invalid_encoding` |
| 2 | **CSV con BOM → descarte silencioso** de todo el manifiesto | la 1ª clave quedaba `'﻿id'`, ningún campo casaba | ignorar en silencio / salida equivocada | leer CSV con `utf-8-sig` |
| 3 | `chunk_chars<1` → `ValueError` sin capturar | `range` con paso 0 | frágil ante lo malformado | degradar al default con aviso |

**El hallazgo 2 es el más peligroso** y el más silencioso: Excel exporta CSV con BOM
por defecto. Un usuario que preparase su manifiesto en Excel habría obtenido **cero
ejemplos sin un solo error** — el peor tipo de fallo, porque parece que "no hay datos"
en vez de "el lector está roto".

**Límite de diseño caracterizado (no un bug puntual):** *streaming abandonado*. Un
cliente que abre un streaming y corta a mitad **no cancela la generación en el
servidor**: llama-cpp sigue produciendo tokens hasta `max_tokens`, reteniendo el
`_INFER_LOCK`, y varios streams abandonados **se serializan y bloquean** a las
peticiones legítimas (medido: un backlog de ~8 streams cortados tardó >90 s en drenar).
**No es un crash ni un deadlock** — el serve se recupera solo. Causa: el
`StreamingResponse` síncrono sobre threadpool no detecta la desconexión hasta que una
escritura al socket falla, y el buffer TCP la retrasa. Fix real = detección de
desconexión asíncrona, que choca con el modelo síncrono + lock → **diferido como cambio
de diseño propio**, no parcheado a las prisas en una sesión de estrés (sería irónico
introducir un bug nuevo arreglando esto).

**Lección de la ronda 3 (sobre el propio proceso):** al fuzzear el Digestor, mi harness
usó `level="auto"` por defecto, detectó el serve real como alcanzable y se puso a
generar Q&A con el LLM para cada chunk de un doc de 2 MB → colgó 5 min golpeando el
serve. *La prueba de estrés se estresó a sí misma.* Y al escribir un test metí un escape
`\ud83d` en un heredoc de shell, que truncó `test_server_oai.py` a 0 bytes (restaurado
con `git checkout`). **Corolario:** el instrumental de la prueba es tan falible como el
sistema probado; git es la red de seguridad que permite ser agresivo sin miedo.

---

## Principios destilados (hasta ahora)

Reglas que estas pruebas han convertido en Reglas de oro del proyecto (ver `SESION.md`):

1. **No acoples un fix a una cadena de error de terceros** sin una lista de marcadores y
   un test con los mensajes reales observados. Los mensajes cambian; el fix no debe morir
   en silencio.
2. **Serializa todo recurso no reentrante** (Regla 19: `_INFER_LOCK`). Bajo un servidor
   concurrente, "funciona en mi prueba de 1 petición" no significa nada.
3. **Valida antes de anunciar.** Si `/health` dice `vision:true`, más vale que una imagen
   funcione. Degradar es honesto; mentir sobre una capacidad no.
4. **Nunca ignores una entrada en silencio.** Una imagen que se perdería → 400 explícito.
   Una salida silenciosamente equivocada es peor que un error ruidoso.
5. **El dato de entrenamiento es sagrado** (Regla 11). Higiene en la frontera: al log
   entra `[imagen]`, nunca un base64; una etiqueta malformada se descarta, no se
   stringifica como basura.
6. **Resiliencia por elemento.** Un lote (manifiesto, carpeta, batch) no debe caer entero
   por un elemento malo. Saltar con reporte claro.
7. **El test se escribe con el dato real del fallo**, después de verlo, no antes con una
   suposición. Si el test puede pasar con el sistema roto, es teatro.
8. **El instrumental de prueba es tan falible como el sistema.** Un harness puede
   golpear el serve sin querer (level=auto), o corromper un fichero (heredoc + unicode).
   Git es lo que permite ser agresivo sin miedo: se rompe, se restaura, se sigue.
9. **Distingue el bug del límite de diseño.** No todo hallazgo se arregla en el momento:
   un crash/deadlock/dato envenenado, sí y ya; una limitación estructural (streaming
   abandonado) se caracteriza, se documenta y se difiere — parchear a las prisas en una
   sesión de estrés arriesga introducir el bug siguiente.

---

## Síntesis: qué nos han enseñado tres rondas

**El número que importa no es "629 tests en verde", es "cuántas familias de fallo hemos
cerrado".** Nueve costuras en tres rondas, y todas caen en siete familias reconocibles
(ver taxonomía). A medida que las cerramos, el estrés tiene que ser más creativo para
encontrar la siguiente — de "2 peticiones lo tumban" (obvio en retrospectiva) a "Excel
mete un BOM que hace descartar todo en silencio" (nada obvio).

**El coste de no estresar es invisible hasta que no lo es.** El crash de concurrencia
llevaba ahí desde que existe el serve; con 1-2 usuarios rara vez coincidían dos
peticiones, así que "funcionaba". La primera vez que Odysseus y Hermes hubieran generado
a la vez en producción, se habría caído — y el síntoma (reinicio de 105 s) habría sido
desconcertante sin este diagnóstico previo.

**La resiliencia es acumulativa y medible.** Cada ronda deja al sistema estrictamente más
duro: ronda 1 arregló la honestidad del multimodal; la 2, la supervivencia bajo
concurrencia; la 3, la higiene de entradas malformadas. El serve de hoy encaja 8
peticiones concurrentes (visión+texto), rechaza limpiamente lo malformado con 4xx
accionables, y ningún lote (manifiesto, batch) cae entero por un elemento malo.

### Pendiente conocido (deuda documentada, no oculta)

- **Streaming abandonado** no cancela la generación server-side (ronda 3). Impacto real
  bajo para 1-2 usuarios; fix = detección de desconexión asíncrona (cambio de diseño).
- **Serve CPU (`:8000`)** corre una imagen anterior a varios fixes; se sanea al
  reconstruirlo.

### Cómo correr una ronda nueva

1. Elige un frente que **no** hayamos apretado (o aprieta más uno viejo).
2. Formula hipótesis hostiles concretas, no "a ver si aguanta".
3. Mide contra el serve REAL (levanta el perfil que toque; recuerda restaurar el 64K).
4. Por cada costura: causa raíz → familia → fix mínimo o "diferido" → test con dato real.
5. Re-verifica el ataque contra el sistema arreglado.
6. Añade la fila a la tabla de su ronda y, si aparece una familia nueva, a la taxonomía.

*Última actualización: 17 jul 2026 — tras la ronda 3.*
