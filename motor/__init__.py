"""
motor
=====
Motor de LoRAs multimodal — interfaz pública del paquete.

Los módulos que dependen de torch/transformers se importan de forma
lazy para que DataDigestor pueda usarse sin GPU ni torch instalado.
"""

# Siempre disponibles (solo requieren pandas / pathlib / json / urllib)
from motor.digestor  import DataDigestor
from motor.analyzer  import ModelAnalyzer
# Detección de hardware — no requiere torch, funciona en todos los entornos
from motor.hardware  import detect_hardware, HardwareProfile

# Módulos que requieren torch/transformers (importación lazy)
def __getattr__(name):
    _torch_modules = {
        "LLMTrainer":       ("motor.trainer_llm",      "LLMTrainer"),
        "VLMTrainer":       ("motor.trainer_vlm",      "VLMTrainer"),
        "ExportManager":    ("motor.exporter",         "ExportManager"),
        "run_server":       ("motor.server",           "run_server"),
        "load_model":       ("motor.server",           "load_model"),
        "LoRAAgent":        ("motor.agent",            "LoRAAgent"),
        "DEFAULT_TOOLS":    ("motor.agent",            "DEFAULT_TOOLS"),
        "ContinualLearner": ("motor.continual",        "ContinualLearner"),
        "run_cycle":        ("motor.continual_cycle",  "run_cycle"),
        "run_benchmark":    ("motor.benchmark_worker", "run_benchmark"),
    }
    if name in _torch_modules:
        module_path, attr = _torch_modules[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'motor' has no attribute {name!r}")

__all__ = [
    "DataDigestor",
    "ModelAnalyzer",
    "detect_hardware",
    "HardwareProfile",
    "LLMTrainer",
    "VLMTrainer",
    "ExportManager",
    "run_server",
    "load_model",
    "LoRAAgent",
    "DEFAULT_TOOLS",
    "ContinualLearner",
    "run_cycle",
    "run_benchmark",
]
