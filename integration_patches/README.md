# Parches de integración

Cambios que aplicamos a dependencias externas (el submódulo `odysseus/`) para
que la integración con el Motor funcione. Viven aquí, en **nuestro** repo,
porque las modificaciones del submódulo se pierden si alguien hace
`git submodule update --force` o reclona — este `.patch` permite reaplicarlas.

## odysseus_agent_loop.patch

**Qué arregla** (descubierto en vivo el 12-jun-2026): Odysseus no enviaba las
herramientas a Gemma 4, así que el modelo respondía de memoria en vez de usar
`web_search` / `trigger_research`. Dos causas, ambas en `src/agent_loop.py`:

1. **Lookup de `supports_tools` por URL que nunca casaba.** Las sesiones de
   chat guardan la URL completa (`.../v1/chat/completions`) pero las fichas de
   endpoint guardan la base (`.../v1`); el código comparaba literal. El parche
   normaliza el sufijo antes de comparar.
2. **Heurística de modelos con function-calling sin Gemma.** La lista de
   modelos "que soportan tools" es anterior a Gemma 4 (y "gemini" no casa con
   "gemma"). El parche añade `gemma-4`/`gemma4`.

> Nota: el problema fue SIEMPRE de Odysseus decidiendo si mandar las tools.
> Nuestro endpoint las maneja bien cuando llegan — verificado por el probe
> (`tests/probe_capacidades.py`, 0 FAIL) y por el test de regresión
> `TestOdysseusIntegrationShape` en `tests/test_server_oai.py`.

### Segundo cambio (29-jun-2026): `max_pages` 5 → 2 en `src/tool_execution.py`

El `web_search` del agente descargaba **5 páginas web completas por búsqueda**.
En una pregunta multi-parte (3 búsquedas × 5 páginas = hasta 15 páginas enteras)
el contexto de la ronda de síntesis se disparaba y un 12B local (Gemma 4 en
RTX 4080) tardaba minutos o se quedaba colgado en el prefill. Bajado a 2: los
snippets ya traen los datos clave; 2 páginas dan profundidad sin desbordar.
Incluido en el mismo `odysseus_agent_loop.patch`.

**Reaplicar** (si el submódulo se resetea):

```bash
cd odysseus
git apply ../integration_patches/odysseus_agent_loop.patch
# luego rebuild de la imagen:
docker compose -f ../docker-compose.unificado.yml build odysseus
```

**Candidato a PR upstream** (apexEvan/odysseus, MIT) — ambos son fixes
genéricos, no específicos de nuestro proyecto.
