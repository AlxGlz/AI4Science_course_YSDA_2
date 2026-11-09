"""Демонстрация пойманной ошибки: 320 -> 250 при переводе отсчётов во время.

Вносит в ``src/ppg/measure.py`` намеренную ошибку — делит номера отсчётов на
250 вместо 320 при расчёте ``t1`` и ``t2`` — и сохраняет unified diff.
Верификатор обязан упасть на проверке известного смещения пика
(``test_known_time_shift``) и на контрольных сигналах §5.

Команды (из корня проекта):
    .\\.venv\\Scripts\\python.exe checks\\evidence\\inject_320_to_250.py --apply
    .\\.venv\\Scripts\\python.exe verify.py
    .\\.venv\\Scripts\\python.exe checks\\evidence\\inject_320_to_250.py --revert

Данные: не читает ``data/``.
Проверка: вывод ``verify.py`` до и после исправления сохранён в
``results/verify/``.
Используется в: ``DIARY.md``, ``ATTEMPTS.md``.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "src" / "ppg" / "measure.py"
EVIDENCE = ROOT / "results" / "verify"

ORIGINAL_1 = "t1=None if p1 is None else 1000.0 * (p1 - s) / fs,"
BROKEN_1 = "t1=None if p1 is None else 1000.0 * (p1 - s) / 250.0,"
ORIGINAL_2 = "t2=None if p2 is None else 1000.0 * (p2 - s) / fs,"
BROKEN_2 = "t2=None if p2 is None else 1000.0 * (p2 - s) / 250.0,"


def _read() -> str:
    return TARGET.read_text(encoding="utf-8")


def _write(text: str) -> None:
    TARGET.write_text(text, encoding="utf-8", newline="")


def apply_bug() -> int:
    text = _read()
    if ORIGINAL_1 not in text or ORIGINAL_2 not in text:
        print("не найден исходный фрагмент — файл уже изменён или правлен")
        return 2
    broken = text.replace(ORIGINAL_1, BROKEN_1).replace(ORIGINAL_2, BROKEN_2)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    diff = difflib.unified_diff(
        text.splitlines(), broken.splitlines(),
        fromfile="src/ppg/measure.py (до)", tofile="src/ppg/measure.py (с ошибкой)",
        lineterm="", n=2,
    )
    (EVIDENCE / "injected_bug_320_to_250.diff").write_text(
        "\n".join(diff) + "\n", encoding="utf-8")
    _write(broken)
    print("ошибка внесена: t1 и t2 делятся на 250.0 вместо fs (320.0)")
    print(f"diff сохранён: {EVIDENCE / 'injected_bug_320_to_250.diff'}")
    return 0


def revert_bug() -> int:
    text = _read()
    if BROKEN_1 not in text or BROKEN_2 not in text:
        print("фрагмент с ошибкой не найден — нечего откатывать")
        return 2
    restored = text.replace(BROKEN_1, ORIGINAL_1).replace(BROKEN_2, ORIGINAL_2)
    _write(restored)
    print("ошибка откатана: t1 и t2 снова делятся на fs")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true")
    group.add_argument("--revert", action="store_true")
    args = parser.parse_args()
    return apply_bug() if args.apply else revert_bug()


if __name__ == "__main__":
    raise SystemExit(main())
