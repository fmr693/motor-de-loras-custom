"""
motor.hardware
==============
Detección automática de hardware y selección del perfil óptimo de ejecución.

No depende de torch — funciona en todos los entornos:
  • serve   (Python 3.12 / CPU / llama-cpp)
  • worker  (Python 3.11 / CUDA / torch)
  • Windows dev (Python 3.13)

Perfiles de ENTRENAMIENTO (training_profile)
--------------------------------------------
  unsloth      — GPU Ampere+ (compute ≥ 8.0), VRAM ≥ 16 GB
                 Unsloth + bf16 + 4bit, batch=4, seq=2048
  peft_bf16    — GPU Ampere+, VRAM 8-16 GB
                 PEFT+TRL + bf16 + 4bit, batch=2, seq=1024
  peft_fp16    — GPU Pascal/Volta (compute < 8.0), VRAM ≥ 8 GB
                 PEFT+TRL + fp16 + 4bit, batch=2, seq=512
  peft_4bit    — GPU, VRAM 4-8 GB
                 PEFT + 4bit + gradient checkpointing, batch=1, seq=512
  cpu_offload  — GPU < 4 GB o iGPU
                 device_map=cpu, sin cuantización, seq=256
  cpu_high     — Sin GPU, RAM ≥ 32 GB (ej: 64 GB DDR5)
                 Mejor que GPU vieja: CPU fp32, batch=1, seq=256
  cpu_low      — Sin GPU, RAM < 32 GB
                 CPU fp32 mínimo, modelos ≤ 1B, seq=128

Perfiles de INFERENCIA (inference_profile) — para llama-cpp-python
-------------------------------------------------------------------
  gpu_full     — VRAM ≥ 16 GB → n_gpu_layers=-1 (todo en GPU)
  gpu_high     — VRAM 8-16 GB → n_gpu_layers=35 (capas parciales)
  gpu_low      — VRAM 4-8 GB  → n_gpu_layers=16
  cpu_fast     — Sin GPU, RAM ≥ 16 GB → todos los cores, contexto largo
  cpu_minimal  — Sin GPU, RAM < 16 GB → la mitad de cores, contexto reducido

Uso rápido
----------
  from motor.hardware import detect_hardware

  hw = detect_hardware()
  print(hw)                         # resumen legible por humanos

  # Para el servidor (llama-cpp):
  llama_kw = hw.llama_kwargs()      # → dict para Llama(**kw)

  # Para el trainer:
  train_kw = hw.training_kwargs()   # → dict con batch_size, max_seq_length, etc.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Detección de GPU — tres métodos en orden de preferencia
# ─────────────────────────────────────────────────────────────────────────────

def _try_torch_cuda() -> Optional[dict]:
    """Usa torch.cuda si está disponible (método más preciso)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        n_gpus = torch.cuda.device_count()
        gpus = []
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index":         i,
                "name":          props.name,
                "vram_total_mb": props.total_memory // (1024 * 1024),
                "vram_free_mb":  0,           # relleno abajo para gpu 0
                "compute_major": props.major,
                "compute_minor": props.minor,
            })
        if n_gpus > 0:
            free, _ = torch.cuda.mem_get_info(0)
            gpus[0]["vram_free_mb"] = free // (1024 * 1024)
        return {"gpus": gpus, "source": "torch"}
    except Exception:
        return None


def _try_pynvml() -> Optional[dict]:
    """Usa pynvml (nvidia management library) si está instalado."""
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(n):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name   = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "index":         i,
                "name":          name,
                "vram_total_mb": mem.total // (1024 * 1024),
                "vram_free_mb":  mem.free  // (1024 * 1024),
                "compute_major": 0,
                "compute_minor": 0,
            })
        pynvml.nvmlShutdown()
        return {"gpus": gpus, "source": "pynvml"}
    except Exception:
        return None


def _try_nvidia_smi() -> Optional[dict]:
    """Fallback: parsea la salida de nvidia-smi (solo requiere el driver)."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        gpus = []
        for i, line in enumerate(result.stdout.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                gpus.append({
                    "index":         i,
                    "name":          parts[0],
                    "vram_total_mb": int(parts[1]),
                    "vram_free_mb":  int(parts[2]),
                    "compute_major": 0,
                    "compute_minor": 0,
                })
            except ValueError:
                continue
        return {"gpus": gpus, "source": "nvidia-smi"} if gpus else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Detección de CPU / RAM
# ─────────────────────────────────────────────────────────────────────────────

def _detect_ram_gb() -> Tuple[float, float]:
    """Devuelve (ram_total_gb, ram_free_gb). Múltiples métodos de fallback."""
    # 1. psutil
    try:
        import psutil
        m = psutil.virtual_memory()
        return m.total / 1e9, m.available / 1e9
    except ImportError:
        pass

    # 2. Linux /proc/meminfo
    try:
        mem_path = Path("/proc/meminfo")
        if mem_path.exists():
            data = mem_path.read_text()
            total_kb = int(re.search(r"MemTotal:\s+(\d+)\s+kB", data).group(1))
            avail_kb = int(re.search(r"MemAvailable:\s+(\d+)\s+kB", data).group(1))
            return total_kb / 1e6, avail_kb / 1e6
    except Exception:
        pass

    # 3. Windows wmic
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "OS", "get",
                 "TotalVisibleMemorySize,FreePhysicalMemory", "/VALUE"],
                capture_output=True, text=True, timeout=5,
            )
            m_total = re.search(r"TotalVisibleMemorySize=(\d+)", result.stdout)
            m_free  = re.search(r"FreePhysicalMemory=(\d+)",     result.stdout)
            if m_total and m_free:
                return int(m_total.group(1)) / 1e6, int(m_free.group(1)) / 1e6
    except Exception:
        pass

    # 4. Fallback conservador (asume 8 GB)
    return 8.0, 4.0


def _detect_cpu_cores() -> Tuple[int, int]:
    """Devuelve (núcleos_físicos, núcleos_lógicos)."""
    logical = os.cpu_count() or 2
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or max(1, logical // 2)
    except ImportError:
        physical = max(1, logical // 2)
    return physical, logical


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de clasificación
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_compute_capability(gpu_name: str) -> Tuple[int, int]:
    """
    Estima la compute capability desde el nombre cuando torch/pynvml no están
    disponibles (solo nvidia-smi o detección externa).
    """
    name = gpu_name.upper()
    # Hopper
    if any(x in name for x in ["H100", "H200", "H800"]):
        return 9, 0
    # Ada Lovelace (RTX 40xx)
    if any(x in name for x in ["RTX 40", "L40", "L4 "]):
        return 8, 9
    # Ampere (RTX 30xx, A100, A30, A40, A10, A6000…)
    if any(x in name for x in ["RTX 30", "A100", "A10", "A30",
                                 "A40", "A6000", "A5000", "A4000", "RTX A"]):
        return 8, 0
    # Turing (RTX 20xx, GTX 1660, T4)
    if any(x in name for x in ["RTX 20", "GTX 1660", "GTX 1650", "T4 ",
                                 "QUADRO RTX"]):
        return 7, 5
    # Volta (V100, Titan V)
    if any(x in name for x in ["V100", "TITAN V"]):
        return 7, 0
    # Pascal (GTX 10xx, P100)
    if any(x in name for x in ["GTX 10", "P100", "P40", "P4 ",
                                 "TITAN X", "TITAN XP"]):
        return 6, 1
    # Maxwell (GTX 9xx, GTX 750)
    if any(x in name for x in ["GTX 9", "GTX 750", "M40", "M60"]):
        return 5, 2
    # Desconocida NVIDIA → asumimos Pascal como mínimo seguro
    return 6, 0


def _is_integrated_gpu(gpu_name: str) -> bool:
    """True si parece una iGPU (Intel HD/UHD/Iris/Arc, AMD Vega iGPU…)."""
    name = gpu_name.upper()
    return any(k in name for k in [
        "INTEL HD", "INTEL UHD", "INTEL IRIS", "INTEL ARC",
        "AMD RADEON VEGA", "RADEON VEGA", "RADEON RX VEGA",
        "RADEON GRAPHICS",   # AMD APU genérico
        "APPLE M",           # MPS, no CUDA
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass principal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    """
    Perfil de hardware detectado + métodos para obtener configuraciones óptimas.
    """
    # ── GPU ──────────────────────────────────────────────────────────────────
    cuda_available:  bool
    n_gpus:          int
    gpu_name:        str
    vram_total_gb:   float
    vram_free_gb:    float
    compute_major:   int
    compute_minor:   int
    is_integrated:   bool

    # ── CPU / RAM ────────────────────────────────────────────────────────────
    cpu_physical:    int
    cpu_logical:     int
    ram_total_gb:    float
    ram_free_gb:     float

    # ── Entorno ──────────────────────────────────────────────────────────────
    os_name:         str    # "windows" / "linux" / "darwin"
    in_docker:       bool
    detect_source:   str    # "torch" / "pynvml" / "nvidia-smi" / "cpu-only"

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades derivadas
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def has_usable_gpu(self) -> bool:
        """GPU CUDA útil: no iGPU y VRAM ≥ 4 GB."""
        return (
            self.cuda_available
            and not self.is_integrated
            and self.vram_total_gb >= 4.0
        )

    @property
    def training_profile(self) -> str:
        """
        Perfil de entrenamiento recomendado según el hardware detectado.

        Lógica de decisión (de mejor a peor):
          GPU Ampere+ ≥ 16 GB → unsloth
          GPU Ampere+ 8-16 GB → peft_bf16
          GPU Pascal/Volta ≥ 8 GB → peft_fp16
          GPU 4-8 GB (cualquier gen) → peft_4bit
          GPU < 4 GB / iGPU → cpu_offload
          Sin GPU, RAM ≥ 32 GB → cpu_high  (ej: 64 GB DDR5 > GPU vieja)
          Sin GPU, RAM < 32 GB → cpu_low
        """
        if self.has_usable_gpu:
            maj  = self.compute_major
            vram = self.vram_total_gb
            if maj >= 8 and vram >= 16:
                return "unsloth"
            if maj >= 8 and vram >= 8:
                return "peft_bf16"
            if vram >= 8:
                return "peft_fp16"
            if vram >= 4:
                return "peft_4bit"
            # GPU presente pero < 4 GB útil
            return "cpu_offload"

        # Sin GPU útil: usamos CPU+RAM
        # 32 GB es el umbral donde CPU supera a una iGPU / GPU con < 2 GB VRAM
        return "cpu_high" if self.ram_total_gb >= 32.0 else "cpu_low"

    @property
    def inference_profile(self) -> str:
        """
        Perfil de inferencia para llama-cpp-python.

        La clasificación depende de VRAM disponible (no solo total) para
        decidir cuántas capas del transformer caben en GPU.
        """
        if self.has_usable_gpu:
            # Usamos vram_free para decisiones de capas (vram_total puede estar ocupada)
            vram = self.vram_free_gb if self.vram_free_gb > 0 else self.vram_total_gb
            # Umbral 12 GB libres (no 14): en Windows/WSL el escritorio reserva
            # ~2.4 GB y una RTX 4080 (16 GB) reportaba 13.6 libres → caía a
            # gpu_high (35 capas, 8.6 tok/s) con un 12B Q4_K_M que necesita
            # ~11 GB con KV de 8k. 12 GB libres bastan para offload completo.
            if vram >= 12:
                return "gpu_full"
            if vram >= 7:
                return "gpu_high"
            return "gpu_low"

        # CPU: umbral 16 GB por el contexto largo que llama.cpp necesita
        return "cpu_fast" if self.ram_total_gb >= 16.0 else "cpu_minimal"

    # ─────────────────────────────────────────────────────────────────────────
    # Configuraciones concretas para cada subsistema
    # ─────────────────────────────────────────────────────────────────────────

    def training_kwargs(self) -> Dict[str, Any]:
        """
        Devuelve un dict con los parámetros recomendados para LLMTrainer.

        Las claves coinciden con los parámetros de LLMTrainer.__init__() y
        LLMTrainer.fit():
          batch_size, grad_accum, max_seq_length, load_in_4bit,
          use_bf16, gradient_checkpointing, use_cpu, recommended_model
        """
        p = self.training_profile

        base: Dict[str, Any] = {
            "hardware_profile":       p,
            "gradient_checkpointing": False,
            "use_cpu":                False,
            "cpu_offload":            False,
        }

        if p == "unsloth":
            base.update(
                batch_size=4, grad_accum=4,
                max_seq_length=2048, load_in_4bit=True, use_bf16=True,
            )
        elif p == "peft_bf16":
            base.update(
                batch_size=2, grad_accum=8,
                max_seq_length=1024, load_in_4bit=True, use_bf16=True,
            )
        elif p == "peft_fp16":
            base.update(
                batch_size=2, grad_accum=8,
                max_seq_length=512, load_in_4bit=True, use_bf16=False,
            )
        elif p == "peft_4bit":
            base.update(
                batch_size=1, grad_accum=8,
                max_seq_length=512, load_in_4bit=True, use_bf16=False,
                gradient_checkpointing=True,
            )
        elif p == "cpu_offload":
            base.update(
                batch_size=1, grad_accum=16,
                max_seq_length=256, load_in_4bit=False, use_bf16=False,
                gradient_checkpointing=True, cpu_offload=True,
            )
        elif p == "cpu_high":
            # 64 GB DDR5 — puede entrenar modelos pequeños en CPU
            # (muy lento pero funcional para fine-tuning incremental de 0.5B)
            base.update(
                batch_size=1, grad_accum=16,
                max_seq_length=256, load_in_4bit=False, use_bf16=False,
                gradient_checkpointing=True, use_cpu=True,
                recommended_model="Qwen/Qwen2.5-0.5B-Instruct",
            )
        else:  # cpu_low
            base.update(
                batch_size=1, grad_accum=32,
                max_seq_length=128, load_in_4bit=False, use_bf16=False,
                gradient_checkpointing=True, use_cpu=True,
                recommended_model="Qwen/Qwen2.5-0.5B-Instruct",
            )
        return base

    def llama_kwargs(self) -> Dict[str, Any]:
        """
        Devuelve un dict con los parámetros para Llama() de llama-cpp-python.

        Claves: n_gpu_layers, n_threads, n_ctx, use_mmap, type_k, type_v, verbose

        type_k / type_v = 8  →  KV cache en Q8_0 (int8).
          - Reduce a la mitad el consumo de RAM del contexto frente a float16.
          - Acelera el prefill en contextos largos (bucles ReAct de 5-8 pasos).
          - Pérdida de calidad < 0.5% en benchmarks estándar de llama.cpp.
          - Si la versión instalada de llama-cpp-python no acepta el parámetro,
            lo ignora silenciosamente (sin romper nada).

        n_threads adaptativo:
          - Se mide la carga actual de CPU en el momento de inicializar.
          - Si la CPU ya supera el 60% de uso (otras apps de oficina activas),
            se reduce n_threads a la mitad para no congelar la máquina.
          - Si psutil no está disponible, se usa el valor estático calculado.
        """
        p       = self.inference_profile
        n_thr   = max(1, self.cpu_logical)
        n_thr_h = max(1, self.cpu_logical // 2)

        # ── Ajuste adaptativo por carga actual de CPU ──────────────────────
        # Mide la carga real del host para no congelar la ofimática del usuario.
        # Intervalo de muestreo = 0.5s (equilibrio entre precisión y latencia de arranque).
        try:
            import psutil as _psutil
            cpu_pct = _psutil.cpu_percent(interval=0.5)
            if cpu_pct > 60:
                # CPU ocupada: ceder hilos al sistema operativo
                n_thr   = max(1, n_thr   // 2)
                n_thr_h = max(1, n_thr_h // 2)
                print(f"  [llama] CPU al {cpu_pct:.0f}% → n_threads reducido "
                      f"a {n_thr} para no interferir con otras apps")
        except Exception:
            pass  # psutil no disponible → usar valor estático
        # ──────────────────────────────────────────────────────────────────

        # KV cache Q8_0 desactivado: type_k=8/type_v=8 no soportado
        # por llama-cpp-python < 0.3.25. Se puede reactivar cuando se actualice.

        # ── Detectar si llama-cpp soporta GPU ──────────────────────────
        # Si el wheel instalado es CPU-only, n_gpu_layers > 0 crashea con
        # Windows Error 0xc000001d (STATUS_ILLEGAL_INSTRUCTION).
        # Detectamos soporte real y forzamos CPU si no hay CUDA en llama.cpp.
        _gpu_ok = False
        try:
            from llama_cpp import llama_supports_gpu_offload
            _gpu_ok = llama_supports_gpu_offload()
        except Exception:
            pass
        if not _gpu_ok:
            p = "cpu_fast"  # forzar CPU aunque detecte GPU NVIDIA
        # ────────────────────────────────────────────────────────────────

        if p == "gpu_full":
            # n_ctx=16384: en modo agéntico, Odysseus mete su system prompt
            # (16 tools) + resultados de búsqueda (15 fragmentos web) en la
            # ronda de síntesis. Con 8192 eso desbordaba y llama.cpp truncaba
            # el prompt → respuesta final vacía (verificado 27-jun-2026, prueba
            # de la pregunta de 3 partes). 16k cabe en la RTX 4080 16 GB
            # (~11,5 GB: pesos 7,7 + KV ~3 + overhead) y absorbe el agente.
            return dict(n_gpu_layers=-1, n_threads=n_thr_h,
                        n_ctx=16384, use_mmap=True, verbose=False)
        if p == "gpu_high":
            return dict(n_gpu_layers=35, n_threads=n_thr_h,
                        n_ctx=4096, use_mmap=True, verbose=False)
        if p == "gpu_low":
            return dict(n_gpu_layers=16, n_threads=n_thr_h,
                        n_ctx=2048, use_mmap=True, verbose=False)
        if p == "cpu_fast":
            # CPU con RAM suficiente pero sin GPU.
            # Usamos la mitad de los cores lógicos para no saturar el bus de memoria.
            # n_ctx=4096 para conversaciones largas (~30-40 mensajes sin truncar).
            return dict(n_gpu_layers=0, n_threads=n_thr_h,
                        n_ctx=4096, use_mmap=True, verbose=False)
        # cpu_minimal
        return dict(n_gpu_layers=0, n_threads=max(2, n_thr_h),
                    n_ctx=2048, use_mmap=True, verbose=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Representación humana
    # ─────────────────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        SEP = "─" * 54
        lines = [
            f"┌{SEP}",
            f"│  Hardware detectado",
            f"├{SEP}",
        ]
        if self.cuda_available:
            lines += [
                f"│  GPU:     {self.gpu_name}",
                f"│  VRAM:    {self.vram_total_gb:.1f} GB total "
                f"| {self.vram_free_gb:.1f} GB libre",
                f"│  Compute: {self.compute_major}.{self.compute_minor}"
                f"  ({'iGPU' if self.is_integrated else 'CUDA OK'})",
            ]
        else:
            lines.append(f"│  GPU:     (no disponible)")
        lines += [
            f"│  CPU:     {self.cpu_physical}c / {self.cpu_logical}t "
            f"| {self.os_name}"
            f"{' · Docker' if self.in_docker else ''}",
            f"│  RAM:     {self.ram_total_gb:.1f} GB total "
            f"| {self.ram_free_gb:.1f} GB libre",
            f"├{SEP}",
            f"│  Perfil entrenamiento : {self.training_profile}",
            f"│  Perfil inferencia    : {self.inference_profile}",
            f"└{SEP}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialización para el endpoint /health."""
        return {
            "cuda_available":   self.cuda_available,
            "n_gpus":           self.n_gpus,
            "gpu_name":         self.gpu_name,
            "vram_total_gb":    self.vram_total_gb,
            "vram_free_gb":     self.vram_free_gb,
            "compute":          f"{self.compute_major}.{self.compute_minor}",
            "is_integrated":    self.is_integrated,
            "cpu_physical":     self.cpu_physical,
            "cpu_logical":      self.cpu_logical,
            "ram_total_gb":     self.ram_total_gb,
            "ram_free_gb":      self.ram_free_gb,
            "os":               self.os_name,
            "in_docker":        self.in_docker,
            "training_profile": self.training_profile,
            "inference_profile": self.inference_profile,
            "detect_source":    self.detect_source,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Función principal — con caché de módulo
# ─────────────────────────────────────────────────────────────────────────────

_CACHED_PROFILE: Optional[HardwareProfile] = None


def detect_hardware(force: bool = False) -> HardwareProfile:
    """
    Detecta el hardware disponible y devuelve un HardwareProfile.

    Resultado cacheado para evitar llamadas repetidas a subprocess.
    Pasa force=True para refrescar (útil tras recargar modelo).

    Orden de detección GPU:
      1. torch.cuda   (más preciso, incluye compute capability)
      2. pynvml       (sin torch, pero con el driver NVML)
      3. nvidia-smi   (solo requiere el driver NVIDIA)
      4. Sin GPU      (modo CPU)
    """
    global _CACHED_PROFILE
    if _CACHED_PROFILE is not None and not force:
        return _CACHED_PROFILE

    # ── GPU ──────────────────────────────────────────────────────────────────
    gpu_info = _try_torch_cuda() or _try_pynvml() or _try_nvidia_smi()

    if gpu_info and gpu_info["gpus"]:
        g              = gpu_info["gpus"][0]
        n_gpus         = len(gpu_info["gpus"])
        gpu_name       = g["name"]
        vram_total_gb  = g["vram_total_mb"] / 1024.0
        vram_free_gb   = g.get("vram_free_mb", g["vram_total_mb"]) / 1024.0
        compute_major  = g["compute_major"]
        compute_minor  = g["compute_minor"]

        # Rellenar compute si no lo tenemos (nvidia-smi no lo da)
        if compute_major == 0:
            compute_major, compute_minor = _estimate_compute_capability(gpu_name)

        cuda_available = True
        is_integrated  = _is_integrated_gpu(gpu_name)
        detect_source  = gpu_info["source"]
    else:
        n_gpus         = 0
        gpu_name       = "CPU"
        vram_total_gb  = 0.0
        vram_free_gb   = 0.0
        compute_major  = 0
        compute_minor  = 0
        cuda_available = False
        is_integrated  = False
        detect_source  = "cpu-only"

    # ── CPU / RAM ─────────────────────────────────────────────────────────────
    cpu_physical, cpu_logical = _detect_cpu_cores()
    ram_total_gb, ram_free_gb = _detect_ram_gb()

    # ── Entorno ───────────────────────────────────────────────────────────────
    os_map  = {"Windows": "windows", "Linux": "linux", "Darwin": "darwin"}
    os_name = os_map.get(platform.system(), platform.system().lower())
    in_docker = (
        Path("/.dockerenv").exists()
        or os.getenv("DOCKER_CONTAINER", "") == "1"
        or os.getenv("container", "") == "docker"
    )

    _CACHED_PROFILE = HardwareProfile(
        cuda_available  = cuda_available,
        n_gpus          = n_gpus,
        gpu_name        = gpu_name,
        vram_total_gb   = round(vram_total_gb, 1),
        vram_free_gb    = round(vram_free_gb, 1),
        compute_major   = compute_major,
        compute_minor   = compute_minor,
        is_integrated   = is_integrated,
        cpu_physical    = cpu_physical,
        cpu_logical     = cpu_logical,
        ram_total_gb    = round(ram_total_gb, 1),
        ram_free_gb     = round(ram_free_gb, 1),
        os_name         = os_name,
        in_docker       = in_docker,
        detect_source   = detect_source,
    )
    return _CACHED_PROFILE


# ─────────────────────────────────────────────────────────────────────────────
# CLI rápido: python -m motor.hardware
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hw = detect_hardware()
    print(hw)
    print()
    print("Kwargs para LLMTrainer:")
    for k, v in hw.training_kwargs().items():
        print(f"  {k:<28} = {v}")
    print()
    print("Kwargs para Llama() (llama-cpp):")
    for k, v in hw.llama_kwargs().items():
        print(f"  {k:<28} = {v}")
