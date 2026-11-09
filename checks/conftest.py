"""Общая настройка проверок: пути импорта и проектный корень.

Данные: нет.
Проверка: сам факт сбора ``verify.py``.
Используется в: всех модулях ``checks/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = Path(__file__).resolve().parent
for path in (str(CHECKS), str(ROOT / "src"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
