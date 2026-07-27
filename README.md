# Motor de LoRAs

> Una fábrica para especializar y desplegar inteligencia artificial en local.
> Convierte cualquier dato (CSV, PDF, imágenes, conversaciones) en un adapter LoRA
> sobre cualquier modelo open-source, y lo sirve de principio a fin. **Sin nube. 0 €/consulta. 100% privado.**

**Resultado que lo demuestra:** partiendo de datos crudos, la fábrica produjo un adapter de 50 MB sobre un modelo de 2B que **supera en 11 puntos de F1 a un ensemble de 6 modelos** en el mismo holdout intocado, y queda a 4 centésimas de un anotador humano ([detalle y metodología](#resultados-medibles)).

**Origen:** pipeline EXIST 2025 (detección de sexismo en memes) generalizado a una fábrica reutilizable.
**Estado:** 665 tests · 0 fallos · 15 comandos CLI · Docker (CPU y GPU) · modelo base actual **Gemma 4 12B** a ~50 tok/s en RTX 4080 · contexto configurable hasta 64K (caché KV cuantizable) · RAG verificado · **visión** (imágenes vía mmproj, perfil multimodal) · aprendizaje híbrido (feedback humano + reflexión) · Digestor multi-modo (clasificar/destilar/conocimiento/VLM) con router `--mode auto` en el CLI · compatible con frontends agénticos (Odysseus y Hermes).

---

## Qué hace

El Motor cubre el ciclo completo de vida de un modelo especializado, sin que el usuario necesite saber de machine learning:

```
datos crudos → DataDigestor → dataset JSONL → ModelAnalyzer → LLMTrainer
→ adapter LoRA (~50 MB) → ExportManager → GGUF (~1 archivo) → servidor local
→ uso real (agente, RAG, chat) → feedback → mejora continua
```

No es un entrenador que necesita otro programa para los datos, ni un servidor que necesita otro para entrenar: es **todo en uno**.

---

## Resultados medibles

El Motor nació del pipeline [EXIST 2025](https://github.com/fmr693/EXIST-2025) (detección multimodal de sexismo en memes, shared task de CLEF 2025) — y volvió a él como **caso de estudio medible de punta a punta**: el **Digestor** generó el dataset (`digestor --mode vlm --manifest`, salida verificada **idéntica byte a byte** al script artesanal que sustituye) y el **VLMTrainer** afinó Qwen2-VL-2B (LoRA r=16, bf16, ~50 min en una RTX 4080 con 6 GB de VRAM). Comparación en el **mismo holdout del 15 %** que ningún modelo vio entrenando:

| Sistema (mismo holdout, 607 memes) | F1 macro | F1 YES |
|---|---|---|
| Pipeline clásico (XLM-RoBERTa + ResNet50, ensemble de 6 modelos) | 0.61 | 0.74 |
| Qwen2-VL-2B zero-shot (umbral calibrado) | 0.62 | 0.73 |
| Qwen2-VL-2B + adapter LoRA del Motor (~50 MB) | 0.70 | 0.79 |
| **ídem, con `mask_prompt` + `keep_best`** | **0.72** | **0.83** |
| *referencia: un anotador humano individual* | *0.76* | — |

**Un solo modelo de 2B con un adapter de 50 MB supera a un ensemble de 6 modelos** (+11 puntos de F1 macro), y queda **a ~4 centésimas de un anotador humano medio**. Ese último dato es el que cierra el análisis: en esta tarea el 45,7 % de los memes no tiene consenso entre anotadores, así que el techo no lo pone el modelo sino la ambigüedad del problema.

Las dos últimas filas se separan por dos mejoras del `VLMTrainer` desarrolladas en el proceso — pérdida calculada solo sobre la respuesta (`mask_prompt`) y quedarse con la mejor época en vez de la última (`keep_best`) — que **solo funcionan combinadas**: por separado, una de ellas empeora el resultado.

**Metodología:** el umbral de decisión se calibra siempre en validación y se mide **una sola vez** en un holdout intocado. Esa disciplina evitó un falso positivo real durante el desarrollo: una variante daba +0.03 en validación y −0.02 en holdout. Protocolo y scripts en el [repo EXIST-2025](https://github.com/fmr693/EXIST-2025).

---

## Madurez: qué está verificado (y qué no)

El proyecto no se valida solo con tests unitarios: se somete a **rondas de estrés contra hardware real** (ver [`PRUEBAS_ESTRES.md`](PRUEBAS_ESTRES.md), bitácora de 5 rondas y 7 familias de fallo).

| Área | Estado verificado |
|---|---|
| **Concurrencia** | Inferencia y log serializados. 16 hilos mixtos + 4 imágenes concurrentes → 0 caídas. *Antes del fix, 2 peticiones simultáneas tumbaban el servidor.* |
| **Entradas malformadas** | JSON/tipos → 422; imagen inválida, texto mal codificado, contexto desbordado → **400 accionables**, no 500 opacos. Verificado bajo carga. |
| **Integridad del dato** | Log íntegro con escrituras concurrentes (0 pérdidas). *Antes del fix se perdían 39 de 300 interacciones.* Nunca entra base64 al log. |
| **Ciclo completo** | Dato → dataset → adapter → **métrica en holdout intocado**, cerrado de punta a punta sobre un caso real (ver Resultados). |
| **Resiliencia por lote** | Un elemento corrupto no tumba el lote: manifiesto con líneas rotas, PDF ilegible o librería ausente degradan **con aviso**, nunca en silencio. |

**Límites conocidos, documentados y no ocultos:**
- Un cliente que corta un *streaming* a mitad no cancela la generación en servidor (retiene el turno; el servicio no cae). Fix = detección asíncrona de desconexión, pendiente.
- Docker Desktop en Windows es el eslabón frágil del stack (2 caídas en una sesión de pruebas). Mitigado con un watchdog; la solución de fondo (WSL2 + Docker Engine nativo) está en el roadmap.
- Visión y contexto de 64K son **alternos**, no simultáneos: no caben a la vez en 16 GB de VRAM.

---

## El modelo: agnóstico por diseño

El Motor funciona con cualquier modelo de HuggingFace (Qwen, Llama, Mistral, Phi, Gemma…). Cambiar de modelo es una línea de configuración. El despliegue actual usa **Gemma 4 12B-it** (Apache 2.0, multimodal, fuerte en español y tool-calling nativo), servido en GGUF Q4_K_M (~7,7 GB) sobre GPU.

> El conocimiento acumulado vive en el **dataset**, no en el adapter. Cuando salga un modelo mejor, se reentrena con los mismos datos: el activo perdura.

---

## Arquitectura — `motor/` (20 módulos)

| Módulo | Función |
|---|---|
| `digestor.py` | DataDigestor: 18 formatos → JSONL. Modos: `classify` (etiquetas), `distill` (charlas con IAs → SFT, con higiene), `knowledge` (documento .txt/.md/.pdf/.docx/.html → Q&A, standalone o con LLM opcional), `vlm` (imágenes → multimodal desde manifiesto: etiquetas, prompt por ejemplo y splits). Semáforo de 5 checks. |
| `analyzer.py` | ModelAnalyzer: detecta arquitectura, elige `target_modules` y configuración. |
| `trainer_llm.py` | LLMTrainer: LoRA (PEFT+TRL) sobre cualquier modelo. Soporta `gemma4_unified`. |
| `trainer_vlm.py` | VLMTrainer: igual para modelos visión-lenguaje. |
| `exporter.py` | ExportManager: merge adapter+base → safetensors o GGUF (Q4_K_M). |
| `server.py` | FastAPI: API compatible OpenAI (`/v1/chat/completions`), `/agent`, `/feedback`, streaming, logging. **Visión opcional**: con `MOTOR_MMPROJ` carga el proyector y acepta imágenes (formato visión de OpenAI). |
| `agent.py` | LoRAAgent (ReAct): razonamiento + uso de herramientas. |
| `domestic_tools.py` | Herramientas reales: organizar archivos, correo, calendario, notas, búsqueda, procesos. |
| `continual.py` | ContinualLearner: replay buffer, rollback automático, registro. |
| `continual_cycle.py` | Ciclo autónomo: digest → train → export → benchmark → promote. |
| `benchmark_worker.py` | Valida un GGUF con 5 tareas (umbral 80%) antes de promover. |
| `dpo_trainer.py` | DPOBuilder: interaction_log → pares chosen/rejected → DPO/ORPO. Fusiona feedback humano + reflexión. |
| `reflection.py` | ReflectionJudge: feedback implícito por LLM-juez (relee el log, infiere aciertos/errores). |
| `log_quality.py` | Filtro de calidad del interaction_log (vacíos, basura, truncados, duplicados). |
| `hardware.py` | `detect_hardware()`: perfiles de entrenamiento e inferencia según GPU/CPU. |
| `odysseus_bridge.py` | Puente con Odysseus: MCP tools + CL bridge + watchdog de promoción. |
| `report.py` | Informe HTML con métricas y smoke test. |
| `domestic_dataset_gen.py` | Generador sintético de datasets. |
| `_model_utils.py` | Helpers de cuantización 4-bit y device info. |
| `__init__.py` | Lazy loader de módulos que dependen de torch. |

---

## CLI — `fabrica_loras.py` (15 comandos)

```bash
fabrica_loras digestor   --data datos.csv --task "..." --output dataset.jsonl          # classify
fabrica_loras digestor   --mode distill   --data ./charlas/ --output sft.jsonl          # charlas IA → SFT
fabrica_loras digestor   --mode knowledge --data manual.md --level template --output qa.jsonl  # doc → Q&A
fabrica_loras digestor   --mode auto      --data ./entrada/ --output ds.jsonl        # detecta tipo → modo
fabrica_loras digestor   --mode vlm --manifest memes.jsonl --label-map "1:YES,0:NO" \
                         --prompt-template "Texto: «{text}»\n¿Sexista? YES/NO." --split train --output train.jsonl  # VLM
fabrica_loras analyzer   --model google/gemma-4-12B-it
fabrica_loras train      --model google/... --data dataset.jsonl --output adapters/mi_adapter/
fabrica_loras vlm        --model Qwen/Qwen2-VL-2B --data ... --output ...
fabrica_loras export     --adapter ... --output ... [--format gguf]
fabrica_loras chat       --model adapters/... [--base-model ...]
fabrica_loras serve      --model modelos/mi_modelo.gguf --host 0.0.0.0
fabrica_loras learn      --adapter ... --auto --log logs/interaction_log.jsonl
fabrica_loras reflect    --log logs/interaction_log.jsonl --out datasets/reflection
fabrica_loras dpo        --log logs/interaction_log.jsonl --output ... --base-model ... [--reflection-dir datasets/reflection]
fabrica_loras cycle      [--only-step digest|train|export|benchmark|promote]
fabrica_loras odysseus   [--mcp-only|--cl-only|--watchdog-only]
fabrica_loras convert-dataset --input ... --framework llamafactory|unsloth|axolotl
fabrica_loras info       dataset.jsonl
```

---

## Instalación

```bash
git clone https://github.com/fmr693/motor-de-loras-custom.git
cd motor-de-loras-custom
pip install -e .                 # base (digestor + utilidades)
pip install -e .[serve]          # + servidor (FastAPI, llama-cpp)
pip install -e .[train]          # + entrenamiento (torch, peft, trl)
```

Servir un modelo:

```bash
python fabrica_loras.py serve --model modelos/<tu_modelo>.gguf --host 0.0.0.0 --port 8000
```

---

## Docker

```bash
# Stack producción (CPU)
docker compose up -d

# Stack unificado con Odysseus (6 servicios)
docker compose -f docker-compose.unificado.yml up -d

# Servir en GPU (RTX 4080) — texto, 64K (el driver diario)
docker compose -f docker-compose.unificado.yml --profile gpu up -d motor-serve-gpu

# Servir con VISIÓN (mismo GGUF + mmproj, 16K) — ALTERNO al anterior, no simultáneo
docker compose -f docker-compose.unificado.yml --profile multimodal up -d motor-serve-mm

# Ciclo de entrenamiento (GPU, on-demand)
docker compose -f docker-compose.unificado.yml --profile train up motor-worker
```

| Imagen | Propósito |
|---|---|
| `Dockerfile.serve` | Inferencia GGUF en CPU, hot-reload |
| `Dockerfile.serve-gpu` | Inferencia GGUF en GPU (llama-cpp CUDA) |
| `Dockerfile.train` | Ciclo de mejora continua (entrenamiento) |

---

## Frontends agénticos

El servidor expone una API estilo OpenAI, así que **cualquier frontend compatible** puede usar el Motor como cerebro local. Verificados en vivo dos, complementarios:

- **[Odysseus](https://github.com/apexEvan/odysseus)** — workspace de IA en el navegador (chat, agentes, RAG de documentos, búsqueda web). Aporta la interfaz de "oficina". Tool-calling agéntico y RAG con embeddings locales verificados (nada sale del equipo). Sus parches (reaplicables, candidatos a PR upstream) están en [`integration_patches/`](integration_patches/).
- **[Hermes Agent](https://github.com/nousresearch/hermes-agent)** (Nous Research, MIT) — agente con memoria persistente, *skills* y gateway a mensajería (Telegram, etc.). El "asistente de bolsillo". Conversación, bucle agéntico y **búsqueda web soberana** (vía el SearXNG del stack, fijado con `web.search_backend`) verificados contra Gemma local.

> Nota de contexto: los clientes agénticos meten mucho en el prompt (system + herramientas + memoria). Hermes exige ≥64K tokens; por eso el serve GPU admite `MOTOR_N_CTX` y `MOTOR_KV_TYPE` (caché KV en Q4/Q8) para ampliar contexto sin salirse de la VRAM. Auditado: sin pérdida de calidad ni de recall.

---

## Tests

665 tests · 0 fallos · tres capas (unitaria, integración E2E, comportamiento).

```bash
PYTHONUTF8=1 python -m pytest tests/ -q       # suite completa
python _run_tests.py --dev                     # solo lo que no necesita GPU
```

---

## Estructura del repositorio

```
motor-de-loras-custom/
├── fabrica_loras.py            CLI (15 comandos)
├── motor/                      20 módulos (ver tabla)
├── tests/                      suites pytest + harnesses E2E en vivo
├── scripts/                    operación: backup, watchdog de Docker, chequeo del activo
├── presentacion/               presentaciones HTML (defensa, etc.)
├── integration_patches/        parches al submódulo Odysseus
├── odysseus/                   submódulo (apexEvan/odysseus)
├── Dockerfile.serve / .serve-gpu / .train
├── docker-compose.yml / docker-compose.unificado.yml
├── pyproject.toml
├── README.md / SESION.md       documentación pública / memoria de desarrollo
├── PRUEBAS_ESTRES.md           bitácora de 5 rondas de estrés + rumbo del proyecto
└── legacy/                     histórico (EXIST 2025, backups)
```

> **Operación** (`scripts/`, ver [`scripts/README_scripts.md`](scripts/README_scripts.md)):
> `backup_activo.py` respalda el dato irreemplazable (log, datasets, memoria);
> `docker_watchdog.ps1` relanza Docker Desktop si el engine cae (con pausa manual para
> cuando lo cierras a propósito); `chequeo_activo.py` mide en seco cuánta señal de
> entrenamiento se ha acumulado. Los dos primeros corren como tareas programadas.
> El activo del proyecto es el **dataset** → protegerlo es parte del diseño.

---

## Roadmap

1. **Régimen de uso — acumular el activo.** El uso diario deja señal de entrenamiento en el log. Medible en seco con `python scripts/chequeo_activo.py`. Umbral fijado de antemano: ~1-2k ejemplos SFT limpios o ~300-500 pares de preferencia → primer entrenamiento de comportamiento. *Línea base (jul 2026): 87 ejemplos SFT limpios.*
2. **Endurecer el despliegue.** Evaluar WSL2 + Docker Engine nativo para eliminar la dependencia de Docker Desktop, en vez de vigilarla.
3. **Deuda técnica conocida:** cancelación de *streaming* abandonado (cambio de diseño, no urgente).

---

*Trabajo de Fin de Grado · 2026 · Licencia MIT. Estado y bitácora de desarrollo en [SESION.md](SESION.md).*
