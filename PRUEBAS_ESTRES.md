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

Las cinco rondas confirman lo mismo: **cada vez que apretamos de verdad (E2E, contra el
serve real, con entradas hostiles), aparecen costuras que 600+ tests en verde no veían.**
No porque los tests estén mal, sino porque prueban lo que *imaginamos* que puede fallar.
El estrés prueba lo que *de verdad* falla. Y las costuras se vuelven más sutiles ronda a
ronda: de "2 peticiones lo tumban" a "el log pierde datos bajo concurrencia" a "dedup
tarda 2 h con un corpus real" — cada vez más cerca del propósito del proyecto.

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
| **Frágil ante lo malformado** | una entrada mala tumba todo el lote | 1 línea JSON rota aborta el manifiesto; user_msg no-string | resiliencia por elemento (saltar con reporte) |
| **500 por culpa del cliente** | error del usuario disfrazado de fallo del servidor | imagen inválida, contexto desbordado | traducir a 4xx accionable |
| **Bloqueo del event loop** | `async def` con trabajo síncrono → congela TODO el serve | `/digestor/process` freezaba /health 2 s | endpoints con trabajo pesado = `def` (threadpool) |

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

### Ronda 4 — Afianzar el ecosistema: el dato bajo concurrencia (2 fallos serios)

Contexto: "casi al 100%". Hipótesis dirigidas al **activo del proyecto (el dataset)** y
al pipeline de datos, más un blitz de ecosistema completo y visión hostil a escala.

**Lo que aguantó:**
- **Blitz de ecosistema**: 16 hilos mezclando texto + tools + streaming + feedback +
  overflow + encoding roto + imagen inválida contra el serve real, 481 s → **serve sin
  crash** (RestartCount 0, health 200); los hostiles que pasaron → **400 correctos bajo
  concurrencia** (invalid_image, invalid_encoding, context_length_exceeded).
- **Visión hostil a escala** (perfil mm): imagen 3000×3000 → OK; base64 truncado y
  data-URI vacío → **400 invalid_image**; mime mentiroso (jpeg/png) → decodifica por
  contenido; 12 imágenes en un mensaje → sin crash; **4 imágenes concurrentes → 4/4 200**.
- **Consolidación**: reconstruido el serve CPU (`:8000`) → salda la deuda de la ronda 3
  (overflow ahora **400**, no 500) y recoge `_LOG_LOCK`.

| # | Costura | Causa raíz | Familia | Fix |
|---|---|---|---|---|
| 1 🔴 | **race del log: interacciones PERDIDAS** | `_log_interaction` (append) y `/feedback` (reescritura total) sin lock → la reescritura pisa los appends intermedios | envenenar/perder el dato | `_LOG_LOCK` serializa ambos |
| 2 | `deduplicate()` **O(n²)** → ~2 h con 50k | Jaccard todos-contra-todos en la pasada near-dupe | frágil a escala | `near_dupe_limit` (def 4000); degradar a exacta con aviso |

**El hallazgo 1 es el más grave de las cuatro rondas para el propósito del proyecto:**
el activo duradero es el **dataset**, no el adapter (decisión estratégica). Un log que
pierde interacciones bajo concurrencia corrompe ese activo **en silencio** — medido:
**39 de 300 interacciones perdidas + 1 corrupta** en segundos de concurrencia. Con
Odysseus y Hermes escribiendo a la vez sobre el mismo serve, más los pulgares 👍/👎 de
la UI reescribiendo el fichero, era cuestión de tiempo. Re-verificado E2E tras el fix:
el blitz dejó **412 líneas, 0 corruptas, 0 duplicados, 37 con feedback**.

**El hallazgo 2** es la clase de bug que no se ve hasta que el proyecto tiene éxito: con
datasets de juguete (cientos de ejemplos) `deduplicate()` es instantáneo; el día que se
acumule un corpus real de decenas de miles —justo el objetivo de "la fábrica como
acumulación de conocimiento"— el pipeline se habría colgado ~2 h sin explicación.

**Lección de la ronda 4:** las dos costuras estaban en el camino del **dato**, no de la
inferencia — y el dato es la tesis del proyecto. *Estresa primero lo que más te importa
perder.* Además, a mitad de ronda **Docker Desktop se cayó** (el `npipe` dejó de
responder); se relanzó y se continuó. Otro recordatorio de la ronda 3: la infraestructura
de la prueba falla tanto como el sistema.

### Ronda 5 — El camino del dato hacia el ENTRENAMIENTO (3 fallos + validación de rumbo)

Contexto: doble objetivo —solidez e información de rumbo. Se estresó el pipeline completo
`log → quality → reflexión/DPO → conversores` (lo que desbloquea el primer LoRA real) y
la superficie HTTP nunca tocada.

**Lo que aguantó / validó (señal de rumbo positiva):**
- **El camino híbrido feedback→reflexión→DPO FUNCIONA E2E** con el juez Gemma real: sobre
  una conversación con corrección explícita ("Eso está mal, es 30"), el juez infirió
  turno 1 = **error** (conf 1.0), turno 2 = **acierto**, y produjo el par DPO
  `{rejected: "35", chosen: "30"}`. La maquinaria de "aprender del uso" es real.
- **DPOBuilder** sobre un log hostil (basura inyectada) → 1 par correcto, sin crash.
- **info** y **convert-dataset** (texto) sobre datos reales → OK. `model=null` → 200.

| # | Costura | Causa raíz | Familia | Fix |
|---|---|---|---|---|
| 1 | `log_quality` **lanza** con `user_msg` no-string | `(x or "").strip()` sobre un int/lista | frágil ante lo malformado | `str()` defensivo (contrato "Nunca lanza") |
| 2 | **conversores corrompen datasets VLM en silencio** | stringifican la lista multimodal como repr de Python | ignorar/envenenar en silencio | `_reject_if_multimodal` aborta con aviso |
| 3 | `POST /digestor/process` **congela el serve entero** | `async def` con trabajo síncrono pesado en el event loop | bloqueo del event loop | `def` → threadpool de FastAPI |

**Los tres están en el camino al primer entrenamiento**, el objetivo de rumbo del proyecto:
- El fallo 1 tumbaba `learn --auto` / DPO con un solo dato mal tipado en el log — y el log
  real ya acumula esa basura (lo metieron los blitzes de la ronda 4). El ciclo nocturno de
  continual learning habría muerto en silencio.
- El fallo 2 es doblemente relevante: el **único dataset real medible** que tienes es EXIST
  (VLM), y es justo el que los conversores corrompían. Si hubieras querido entrenarlo en
  Unsloth/LLaMA-Factory, habrías entrenado con basura.
- El fallo 3 es una **familia nueva** (bloqueo del event loop): distinta del crash de
  thread-safety de la ronda 2. Un `async def` que hace trabajo síncrono no rompe, pero
  congela TODO el serve durante el procesado — invisible hasta que dos cosas pasan a la vez.

**Lección de la ronda 5:** al estresar el camino del dato encontramos que **estaba minado
justo donde el proyecto quiere avanzar**. Y una señal de infraestructura que ya es tendencia:
**Docker Desktop se cayó por SEGUNDA vez** esta sesión (rondas 4 y 5). No es anecdótico —
es el eslabón frágil del ecosistema soberano en Windows (ver Rumbo).

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
- ~~Serve CPU (`:8000`) corre una imagen anterior a varios fixes~~ → **saldado en la
  ronda 4** (reconstruido; overflow → 400, `_LOG_LOCK` incluido).

### Estado del ecosistema tras 5 rondas (17 jul 2026)

- **Concurrencia**: inferencia serializada (`_INFER_LOCK`) y log serializado (`_LOG_LOCK`).
  16 hilos mixtos + 4 imágenes concurrentes → 0 crashes.
- **Entradas malformadas**: JSON/tipos → 422; imagen inválida, texto mal codificado,
  contexto desbordado → **400 accionables** (no 500), verificado bajo concurrencia.
- **Integridad del dato**: log íntegro bajo appends+feedback concurrentes (0 pérdidas);
  al log entra `[imagen]`, nunca base64; label malformada se descarta.
- **Visión**: honesta (valida mmproj antes de anunciar); robusta ante imagen grande,
  base64 roto, mime mentiroso, multi-imagen y concurrencia.
- **Digestor**: resiliente por elemento (manifiesto malformado no cae entero); dedup con
  tope O(n²) avisado; BOM de Excel neutralizado; F4 idéntico al script bespoke sobre
  datos reales.
- **Camino al entrenamiento**: log_quality resiliente a basura; conversores rechazan VLM
  con aviso; `/digestor/process` no bloquea el event loop; reflexión→DPO validado E2E.
- **Deuda restante**: streaming abandonado (diseño) + fragilidad de Docker Desktop en
  Windows (2 caídas esta sesión). Ver "Rumbo".

## Rumbo del proyecto (lo que las pruebas revelan sobre hacia dónde ir)

Estresar no solo endurece: **muestra dónde está el proyecto de verdad**. Lo que las 5
rondas dicen sobre el siguiente paso:

1. **El primer entrenamiento real ya está DESBLOQUEADO — y el candidato es EXIST-VLM.**
   La ronda 5 validó los dos extremos: el camino reflexión→DPO produce señal de
   entrenamiento correcta (par chosen/rejected), y F4 genera el dataset VLM idéntico al
   script bespoke. El único dominio **medible** que tienes es EXIST (hay holdout + F1 de
   referencia del pipeline clásico). Rumbo natural: **entrenar el LoRA VLM sobre EXIST con
   el VLMTrainer y comparar F1 contra el holdout** — cierra el círculo "fábrica → adapter
   → métrica" por primera vez de punta a punta. (Ojo: los conversores externos NO sirven
   para VLM —ronda 5—; se entrena con el VLMTrainer del Motor.)

2. **El feedback implícito ya tiene tubería; falta CAUDAL.** El juez funciona, pero el log
   real es pequeño y casi todo aciertos. El valor se acumula con el USO: cuanto más se use
   Gemma vía Odysseus/Hermes, más señal DPO produce la reflexión. Rumbo: **usar el stack a
   diario** es, literalmente, generar el activo. No hay atajo de ingeniería para esto.

3. **La deuda pendiente es de resiliencia, no de features.** Tras 5 rondas, lo que queda
   son dos límites conocidos, ambos de diseño: el **streaming abandonado** (ronda 3) y la
   fragilidad de **Docker Desktop en Windows** (se cayó 2 veces esta sesión). El segundo es
   el más estratégico: el stack soberano depende de un Docker Desktop que no es fiable.
   Rumbo posible: evaluar WSL2 + Docker Engine nativo (sin Desktop) o un arranque
   supervisado, para que la caída de Docker no tumbe el ecosistema entero.

4. **El código está maduro; la superficie ya se sostiene.** Las costuras de la ronda 5 no
   fueron de lógica de negocio rota, sino de robustez en los bordes (tipos, async,
   formatos). Eso sugiere que la base es sólida y el trabajo de valor ya no es "arreglar el
   motor" sino **usarlo para producir el dataset y medir** — que es la tesis original.

**Síntesis de rumbo:** el ecosistema está lo bastante firme para dejar de construir tubería
y empezar a **bombear dato real** por ella. El siguiente hito con más señal sería el primer
LoRA medible (EXIST-VLM), y en paralelo, endurecer el eslabón Docker.

---

### Cómo correr una ronda nueva

1. Elige un frente que **no** hayamos apretado (o aprieta más uno viejo).
2. Formula hipótesis hostiles concretas, no "a ver si aguanta".
3. Mide contra el serve REAL (levanta el perfil que toque; recuerda restaurar el 64K).
4. Por cada costura: causa raíz → familia → fix mínimo o "diferido" → test con dato real.
5. Re-verifica el ataque contra el sistema arreglado.
6. Añade la fila a la tabla de su ronda y, si aparece una familia nueva, a la taxonomía.

*Última actualización: 17 jul 2026 — tras la ronda 5.*
