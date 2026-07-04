"""
motor.report
============
ReportGenerator: genera un informe HTML post-entrenamiento con métricas,
comparativas base vs adapter y recomendaciones.

El informe es standalone (CSS embebido, sin dependencias externas).
Se puede abrir en cualquier navegador o exportar a PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_training_report(
    adapter_dir: str,
    metrics: Dict[str, Any],
    smoke_test: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
) -> Path:
    """
    Genera un informe HTML de entrenamiento en la carpeta del adapter.

    Parámetros
    ----------
    adapter_dir : str
        Carpeta donde se guarda el adapter.
    metrics : dict
        Métricas del entrenamiento (train_loss, eval_loss, elapsed_min, etc.)
    smoke_test : dict, opcional
        Resultados del smoke test (samples, passed, avg_difference_pct).
    domain : str, opcional
        Dominio detectado del dataset.

    Devuelve
    --------
    Path — ruta al archivo training_report.html generado.
    """
    out = Path(adapter_dir) / "training_report.html"

    # ── Extraer datos ──────────────────────────────────────────────
    adapter_name = Path(adapter_dir).name
    model_id = metrics.get("model_id", "unknown")
    train_loss = metrics.get("train_loss", "N/A")
    eval_loss = metrics.get("eval_loss", "N/A")
    elapsed = metrics.get("elapsed_min", 0)
    vram = metrics.get("vram_peak_gb", 0)
    engine = metrics.get("engine", "unknown")
    epochs = metrics.get("epochs", "?")
    lora_r = metrics.get("lora_r", "?")
    lora_alpha = metrics.get("lora_alpha", "?")
    batch = metrics.get("batch_effective", "?")
    train_n = metrics.get("train_samples", "?")
    eval_n = metrics.get("eval_samples", "?")

    # ── Smoke test samples ─────────────────────────────────────────
    smoke_samples = smoke_test.get("samples", []) if smoke_test else []
    smoke_passed = smoke_test.get("passed", None) if smoke_test else None
    smoke_skipped = smoke_test.get("skipped", False) if smoke_test else False
    smoke_diff = smoke_test.get("avg_difference_pct", 0) if smoke_test else 0
    smoke_reason = smoke_test.get("skip_reason", "") if smoke_test else ""

    # ── Generar tabla de ejemplos ──────────────────────────────────
    sample_rows = ""
    for i, s in enumerate(smoke_samples[:5]):
        diff_class = "diff-high" if s.get("difference_pct", 0) > 50 else "diff-med" if s.get("difference_pct", 0) > 20 else "diff-low"
        sample_rows += f"""<tr>
            <td class="example-input">{_esc(s['input'][:100])}</td>
            <td class="example-output">{_esc(s['base_output'][:80])}</td>
            <td class="example-output" style="color:#22c55e">{_esc(s['adapter_output'][:80])}</td>
            <td class="{diff_class}">{s.get('difference_pct', '?')}%</td>
        </tr>"""

    # ── Evaluar calidad ────────────────────────────────────────────
    recommendations = _generate_recommendations(metrics, smoke_test)

    # ── Construir HTML ─────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Training Report — {adapter_name}</title>
<style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;padding:20px}}
    .container{{max-width:900px;margin:0 auto}}
    .header{{background:linear-gradient(135deg,#1e3a8a,#3b82f6);border-radius:12px;padding:32px;margin-bottom:24px;text-align:center}}
    .header h1{{font-size:28px;color:#fff;margin-bottom:8px}}
    .header .subtitle{{color:#93c5fd;font-size:14px}}
    .card{{background:#1a1d2e;border:1px solid #2d3148;border-radius:10px;padding:24px;margin-bottom:20px}}
    .card h2{{font-size:18px;color:#60a5fa;margin-bottom:16px;border-bottom:1px solid #2d3148;padding-bottom:8px}}
    .metrics-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}}
    .metric{{background:#0f1117;border-radius:8px;padding:16px;text-align:center}}
    .metric .value{{font-size:24px;font-weight:800;color:#60a5fa}}
    .metric .label{{font-size:11px;color:#64748b;margin-top:4px;text-transform:uppercase;letter-spacing:.06em}}
    .smoke-status{{padding:10px 16px;border-radius:8px;margin-bottom:16px;font-weight:600}}
    .smoke-ok{{background:#064e3b;color:#6ee7b7;border:1px solid#065f46}}
    .smoke-fail{{background:#7f1d1d;color:#fca5a5;border:1px solid#991b1b}}
    .params-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
    .param{{background:#0f1117;border-radius:6px;padding:12px}}
    .param .pval{{font-weight:700;color:#f8fafc}}
    .param .plbl{{font-size:11px;color:#64748b}}
    table{{width:100%;border-collapse:collapse;font-size:12px}}
    th{{background:#1e293b;padding:10px 12px;text-align:left;font-weight:600;color:#94a3b8;font-size:11px;text-transform:uppercase}}
    td{{padding:10px 12px;border-bottom:1px solid#1e293b;vertical-align:top}}
    .example-input{{color:#93c5fd;max-width:200px;word-break:break-all}}
    .example-output{{max-width:180px;word-break:break-all}}
    .diff-high{{color:#22c55e;font-weight:700}}
    .diff-med{{color:#eab308;font-weight:700}}
    .diff-low{{color:#ef4444;font-weight:700}}
    .recs{{list-style:none}}
    .recs li{{padding:8px 0;border-bottom:1px solid#1e293b;line-height:1.5}}
    .recs li:last-child{{border-bottom:none}}
    .footer{{text-align:center;color:#475569;font-size:11px;padding:20px}}
    @media print{{body{{background:#fff;color:#000}}.card{{background:#fff;border:1px solid#ccc}}.header{{background:#1e3a8a}}}}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>📊 Training Report</h1>
    <div class="subtitle">{adapter_name} — {model_id}</div>
</div>

<div class="card">
    <h2>🧪 Smoke Test</h2>
    {"<div class='smoke-status smoke-ok'>✅ ADAPTER FUNCIONA (diferencia media: " + str(smoke_diff) + "%)</div>" if smoke_passed else "<div class='smoke-status smoke-fail'>❌ ADAPTER POSIBLEMENTE NO ENTRENADO (diferencia media: " + str(smoke_diff) + "% < 10%)</div>" if smoke_passed is False else "<div class='smoke-status' style='background:#1e293b'>⏭ Smoke test omitido" + (": " + smoke_reason if smoke_reason else "") + "</div>" if smoke_skipped else "<div class='smoke-status' style='background:#1e293b'>Sin datos de smoke test</div>"}
    <table>
        <tr><th>Input</th><th>Base Model</th><th>Adapter</th><th>Diff</th></tr>
        {sample_rows or '<tr><td colspan="4" style="color:#64748b">No hay ejemplos disponibles</td></tr>'}
    </table>
</div>

<div class="card">
    <h2>📈 Métricas</h2>
    <div class="metrics-grid">
        <div class="metric"><div class="value">{train_loss}</div><div class="label">Train Loss</div></div>
        <div class="metric"><div class="value">{eval_loss}</div><div class="label">Eval Loss</div></div>
        <div class="metric"><div class="value">{elapsed} min</div><div class="label">Tiempo</div></div>
        <div class="metric"><div class="value">{vram} GB</div><div class="label">VRAM Pico</div></div>
    </div>
    <div class="params-grid">
        <div class="param"><div class="pval">r={lora_r}, α={lora_alpha}</div><div class="plbl">LoRA</div></div>
        <div class="param"><div class="pval">{epochs} épocas</div><div class="plbl">Entrenamiento</div></div>
        <div class="param"><div class="pval">batch {batch}</div><div class="plbl">Batch efectivo</div></div>
        <div class="param"><div class="pval">{train_n} / {eval_n}</div><div class="plbl">Train / Eval samples</div></div>
        <div class="param"><div class="pval">{engine}</div><div class="plbl">Motor</div></div>
        <div class="param"><div class="pval">{domain or 'N/A'}</div><div class="plbl">Dominio</div></div>
    </div>
</div>

<div class="card">
    <h2>💡 Recomendaciones</h2>
    <ul class="recs">{''.join(f'<li>{r}</li>' for r in recommendations)}</ul>
</div>

<div class="footer">
    Generado por Fábrica de LoRAs Especializados — {_now()}
</div>

</div>
</body>
</html>"""

    out.write_text(html, encoding="utf-8")
    print(f"[Report] Informe generado: {out}")
    return out


# ------------------------------------------------------------------
# Helpers internos
# ------------------------------------------------------------------

def _esc(text: str) -> str:
    """Escapa HTML básico."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _now() -> str:
    """Fecha actual formateada."""
    from datetime import datetime
    return datetime.now().strftime("%d de %B de %Y, %H:%M")


def _generate_recommendations(
    metrics: Dict[str, Any],
    smoke_test: Optional[Dict[str, Any]],
) -> List[str]:
    """Genera recomendaciones basadas en las métricas."""
    recs = []

    train_loss = metrics.get("train_loss", 0)
    eval_loss = metrics.get("eval_loss", 0)
    epochs = metrics.get("epochs", 3)
    lora_r = metrics.get("lora_r", 16)
    samples = metrics.get("train_samples", 0)

    # Gap train/eval
    if train_loss and eval_loss:
        gap = eval_loss - train_loss
        if gap > 0.15:
            recs.append(
                f"⚠️ Gap train/eval elevado ({gap:.3f}). "
                f"Posible overfitting. Recomendaciones:<br>"
                f"  • Reducir r (actual={lora_r})<br>"
                f"  • Aumentar datos de entrenamiento (actual={samples})<br>"
                f"  • Usar más dropout o weight decay"
            )
        elif gap < 0.05:
            recs.append(
                f"✅ Gap train/eval bajo ({gap:.3f}). "
                f"El modelo generaliza bien."
            )

    # Smoke test
    if smoke_test:
        if smoke_test.get("skipped"):
            recs.append(
                "⏭ Smoke test omitido — el modelo no es PeftModel. "
                "Solo aplicable a adapters PEFT (LoRA/QLoRA)."
            )
        elif not smoke_test.get("passed", False):
            recs.append(
                "❌ Smoke test NO superado. El adapter no cambia el output. "
                "Verifica: carga del modelo, merge, target_modules, dataset."
            )
    else:
        recs.append(
            "⚠️ No se ejecutó smoke test. Se recomienda verificar "
            "manualmente que el adapter funciona."
        )

    # Pocas muestras
    if samples and samples < 200:
        recs.append(
            f"⚠️ Pocos ejemplos de entrenamiento ({samples}). "
            f"El adapter puede no generalizar bien. Considera data augmentation."
        )
    elif samples and samples > 5000:
        recs.append(
            f"✅ Buen tamaño de dataset ({samples} ejemplos). "
            f"El adapter debería generalizar correctamente."
        )

    # Épocas
    if epochs >= 4 and samples and samples < 1000:
        recs.append(
            f"⚠️ {epochs} épocas con {samples} ejemplos. "
            f"Riesgo de overfitting. Monitoriza eval_loss."
        )

    if not recs:
        recs.append("✅ No se detectaron problemas. El adapter parece entrenado correctamente.")

    return recs
