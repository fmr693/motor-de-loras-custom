# Motor de LoRAs

> Una fábrica para especializar y desplegar inteligencia artificial en local.
> Convierte cualquier dato (CSV, PDF, imágenes, conversaciones) en un adapter LoRA
> sobre cualquier modelo open-source, y lo sirve de principio a fin. **Sin nube. 0 €/consulta. 100% privado.**

**Origen:** pipeline EXIST 2025 (detección de sexismo en memes) generalizado a una fábrica reutilizable.
**Estado:** 511 tests · 0 fallos · 14 comandos CLI · Docker (CPU y GPU) · modelo base actual **Gemma 4 12B** a ~50 tok/s en RTX 4080 · stack con Odysseus integrado y RAG verificado.

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

## El modelo: agnóstico por diseño

El Motor funciona con cualquier modelo de HuggingFace (Qwen, Llama, Mistral, Phi, Gemma…). Cambiar de modelo es una línea de configuración. El despliegue actual usa **Gemma 4 12B-it** (Apache 2.0, multimodal, fuerte en español y tool-calling nativo), servido en GGUF Q4_K_M (~7,7 GB) sobre GPU.

> El conocimiento acumulado vive en el **dataset**, no en el adapter. Cuando salga un modelo mejor, se reentrena con los mismos datos: el activo perdura.

---

## Arquitectura — `motor/` (19 módulos)

| Módulo | Función |
|---|---|
| `digestor.py` | DataDigestor: 18 formatos de entrada → JSONL. Semáforo de 5 checks. Detección de dominio. |
| `analyzer.py` | ModelAnalyzer: detecta arquitectura, elige `target_modules` y configuración. |
| `trainer_llm.py` | LLMTrainer: LoRA (PEFT+TRL) sobre cualquier modelo. Soporta `gemma4_unified`. |
| `trainer_vlm.py` | VLMTrainer: igual para modelos visión-lenguaje. |
| `exporter.py` | ExportManager: merge adapter+base → safetensors o GGUF (Q4_K_M). |
| `server.py` | FastAPI: API compatible OpenAI (`/v1/chat/completions`), `/agent`, `/feedback`, streaming, logging. |
| `agent.py` | LoRAAgent (ReAct): razonamiento + uso de herramientas. |
| `domestic_tools.py` | Herramientas reales: organizar archivos, correo, calendario, notas, búsqueda, procesos. |
| `continual.py` | ContinualLearner: replay buffer, rollback automático, registro. |
| `continual_cycle.py` | Ciclo autónomo: digest → train → export → benchmark → promote. |
| `benchmark_worker.py` | Valida un GGUF con 5 tareas (umbral 80%) antes de promover. |
| `dpo_trainer.py` | DPOBuilder: interaction_log → pares chosen/rejected → DPO/ORPO. |
| `log_quality.py` | Filtro de calidad del interaction_log (vacíos, basura, truncados, duplicados). |
| `hardware.py` | `detect_hardware()`: perfiles de entrenamiento e inferencia según GPU/CPU. |
| `odysseus_bridge.py` | Puente con Odysseus: MCP tools + CL bridge + watchdog de promoción. |
| `report.py` | Informe HTML con métricas y smoke test. |
| `domestic_dataset_gen.py` | Generador sintético de datasets. |
| `_model_utils.py` | Helpers de cuantización 4-bit y device info. |
| `__init__.py` | Lazy loader de módulos que dependen de torch. |

---

## CLI — `fabrica_loras.py` (14 comandos)

```bash
fabrica_loras digestor   --data datos.csv --task "..." --output dataset.jsonl
fabrica_loras analyzer   --model google/gemma-4-12B-it
fabrica_loras train      --model google/... --data dataset.jsonl --output adapters/mi_adapter/
fabrica_loras vlm        --model Qwen/Qwen2-VL-2B --data ... --output ...
fabrica_loras export     --adapter ... --output ... [--format gguf]
fabrica_loras chat       --model adapters/... [--base-model ...]
fabrica_loras serve      --model modelos/mi_modelo.gguf --host 0.0.0.0
fabrica_loras learn      --adapter ... --auto --log logs/interaction_log.jsonl
fabrica_loras dpo        --log logs/interaction_log.jsonl --output ... --base-model ...
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

# Servir en GPU (RTX 4080)
docker compose -f docker-compose.unificado.yml --profile gpu up -d motor-serve-gpu

# Ciclo de entrenamiento (GPU, on-demand)
docker compose -f docker-compose.unificado.yml --profile train up motor-worker
```

| Imagen | Propósito |
|---|---|
| `Dockerfile.serve` | Inferencia GGUF en CPU, hot-reload |
| `Dockerfile.serve-gpu` | Inferencia GGUF en GPU (llama-cpp CUDA) |
| `Dockerfile.train` | Ciclo de mejora continua (entrenamiento) |

---

## Integración con Odysseus

[Odysseus](https://github.com/apexEvan/odysseus) es un workspace de IA (chat, agentes, RAG, búsqueda web). El Motor le aporta el backend de inferencia local; Odysseus aporta la interfaz y las herramientas. Se conectan por la API estilo OpenAI.

Verificado en vivo: tool-calling agéntico (el modelo decide y ejecuta herramientas), búsqueda web, y RAG sobre documentos propios con embeddings locales (nada sale del equipo).

Los parches necesarios al submódulo Odysseus están en [`integration_patches/`](integration_patches/) (reaplicables; candidatos a PR upstream).

---

## Tests

511 tests · 0 fallos · tres capas (unitaria, integración E2E, comportamiento).

```bash
PYTHONUTF8=1 python -m pytest tests/ -q       # suite completa
python _run_tests.py --dev                     # solo lo que no necesita GPU
```

---

## Estructura del repositorio

```
motor-de-loras-custom/
├── fabrica_loras.py            CLI (14 comandos)
├── motor/                      19 módulos (ver tabla)
├── tests/                      suites pytest + harnesses E2E en vivo
├── presentacion/               presentaciones HTML (defensa, etc.)
├── integration_patches/        parches al submódulo Odysseus
├── odysseus/                   submódulo (apexEvan/odysseus)
├── Dockerfile.serve / .serve-gpu / .train
├── docker-compose.yml / docker-compose.unificado.yml
├── pyproject.toml
├── README.md / SESION.md       documentación pública / memoria de desarrollo
└── legacy/                     histórico (EXIST 2025, backups)
```

---

## Origen académico — EXIST 2025 (el círculo cerrado)

El Motor nació del pipeline [EXIST 2025](https://github.com/fmr693/EXIST-2025) (detección multimodal de sexismo en memes, shared task de CLEF 2025) — y en julio de 2026 volvió a él como **primer caso de estudio medible**: el `VLMTrainer` afinó Qwen2-VL-2B (LoRA r=16, bf16, 3 épocas, **52 min en una RTX 4080 con 6 GB de VRAM**) y se comparó con el pipeline clásico original en el **mismo holdout del 15%** que ningún modelo vio entrenando:

| Sistema (mismo holdout, 607 memes) | F1 macro | F1 YES |
|---|---|---|
| Pipeline clásico (XLM-RoBERTa + ResNet50, ensemble de 6 modelos) | 0.61 | 0.74 |
| Qwen2-VL-2B zero-shot (umbral calibrado) | 0.62 | 0.73 |
| **Qwen2-VL-2B + adapter LoRA del Motor (~50 MB)** | **0.70** | **0.79** |

Un solo modelo de 2B con un adapter del Motor supera al ensemble completo: **+8,7 puntos de F1 macro**. Protocolo, scripts y detalles en el [repo EXIST-2025](https://github.com/fmr693/EXIST-2025).

---

*Trabajo de Fin de Grado · 2026 · Licencia MIT. Estado y bitácora de desarrollo en [SESION.md](SESION.md).*
