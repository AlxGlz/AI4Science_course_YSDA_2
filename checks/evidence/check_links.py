"""Проверка относительных ссылок и путей в markdown-файлах проекта.

Все относительные цели `[текст](путь)` в README.md, REPORT.md, SUBGOALS.md,
DIARY.md, ATTEMPTS.md и METHOD.md должны указывать на существующие файлы
или папки. Внешние ссылки и якоря пропускаются.

Данные: markdown-файлы в корне проекта.
Проверка: запуск из корня; код 0 — битых ссылок нет, код 1 — есть.
Используется в: ручной проверке документации перед сдачей; в верификатор
не входит, потому что зависит от расположения файлов, а не от расчётов.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["README.md", "REPORT.md", "SUBGOALS.md", "DIARY.md", "ATTEMPTS.md", "METHOD.md"]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    bad: list[tuple[str, str]] = []
    total = 0
    for name in FILES:
        path = ROOT / name
        if not path.exists():
            bad.append((name, "<файл отсутствует>"))
            continue
        text = path.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            total += 1
            clean = target.split("#", 1)[0]
            if not (ROOT / clean).exists():
                bad.append((name, target))

    print(f"Ссылок проверено: {total}; битых: {len(bad)}")
    for name, target in bad:
        print(f"  {name}: {target}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
