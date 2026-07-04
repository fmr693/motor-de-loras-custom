"""
tests/conftest.py
=================
Configuración global de pytest para el proyecto.

Añade la raíz del proyecto al sys.path una sola vez, de forma que
todos los tests puedan importar `motor` sin necesidad de `sys.path.insert`
manual en cada archivo.

También define fixtures compartidas entre test files.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Garantiza que `import motor` funciona desde cualquier test,
# independientemente de desde dónde se lance pytest.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
