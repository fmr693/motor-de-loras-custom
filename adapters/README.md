# adapters/ — adaptadores LoRA y artefactos de terceros

Cada subcarpeta es un adaptador entrenado por este proyecto. Junto a los pesos se
guardan los artefactos del **modelo base** necesarios para servirlo, que **no son
obra de este proyecto** y conservan la licencia de origen.

## Qué hay en cada carpeta

| Fichero | Origen | Licencia |
|---|---|---|
| `adapter_config.json`, `adapter_model.safetensors` | Este proyecto (entrenado) | MIT, como el resto del repo |
| `meta.json`, `training_report.html` | Este proyecto (generado) | MIT |
| `README.md` | Plantilla de tarjeta de modelo de PEFT/`trl` | Apache 2.0 (Hugging Face) |
| `tokenizer/tokenizer.json` | **Modelo base Qwen2.5** | La del modelo base (ver abajo) |
| `tokenizer/tokenizer_config.json` | **Modelo base Qwen2.5** | La del modelo base |
| `tokenizer/chat_template.jinja` | **Modelo base Qwen2.5** | La del modelo base |

Los `.safetensors` no se versionan (ver `.gitignore`): se distribuyen aparte.

## Atribución del tokenizer

Los ficheros de `tokenizer/` se redistribuyen **sin modificar** desde los modelos
base de la familia **Qwen2.5**, de Alibaba Cloud / Qwen Team. No son obra de este
proyecto y la licencia MIT del repositorio **no les aplica**.

| Adaptador | Modelo base | Licencia declarada por el modelo base |
|---|---|---|
| `domestic_7b` | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `apache-2.0` |
| `finance_sentiment_14b` | [`Qwen/Qwen2.5-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | `apache-2.0` |
| `domestic_base` | [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | **`qwen-research`** — no es Apache 2.0 |
| `domestic_v2` | [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | **`qwen-research`** — no es Apache 2.0 |

> **Aviso.** La familia Qwen2.5 es `apache-2.0` **salvo** las variantes 3B y 72B, que
> se publican bajo licencias propias. La 3B usa `qwen-research`, que **restringe el uso
> comercial**. Afecta a `domestic_base` y `domestic_v2`, y por herencia a los
> adaptadores entrenados sobre ellos.
>
> Los tres identificadores se verificaron contra las tarjetas de modelo en Hugging
> Face el **1 de septiembre de 2026**. La referencia válida sigue siendo la tarjeta
> del modelo, que su autor puede cambiar.

Un adaptador LoRA no contiene pesos del modelo base, pero solo tiene sentido
aplicado sobre él: las condiciones del modelo base gobiernan su uso.

## Otros artefactos de terceros en el repositorio

- **`integration_patches/odysseus_agent_loop.patch`** — incluye fragmentos de código
  de [Odysseus](https://github.com/apexEvan/odysseus) en el contexto del diff. El
  parche se distribuye para reaplicar cambios sobre ese proyecto; el código citado
  pertenece a sus autores y conserva su licencia.
- **`odysseus/`** — submódulo Git: es una referencia a un commit del repositorio
  original, no una copia de su código.
- **`config/searxng/settings.yml`** — configuración propia sobre
  [SearXNG](https://github.com/searxng/searxng) (AGPL-3.0) mediante
  `use_default_settings: true`. No redistribuye código de SearXNG.
