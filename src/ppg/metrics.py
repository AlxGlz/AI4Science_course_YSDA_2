"""Ошибки измерения и усреднение по §2 методики.

Нормированные ошибки (в процентах) считаются для каждой волны до усреднения:

* ``A1`` — на ``A1`` эталона;
* ``t1``, ``t2`` — на длительность волны ``T``;
* ``A2`` — на ``A1`` эталона (не на собственную амплитуду второго пика).

Усреднение — последовательное арифметическое с равными весами: сначала по
волнам одной реализации, затем по реализациям одного эталона, затем по
эталонам. Пустые группы пропускаются и не заменяются нулём. Отказы
учитываются отдельно, а не как нулевая ошибка.

Данные: значения ошибок и метки групп; файлы не читаются.
Проверка: ``checks/test_metrics.py`` (усреднение 2×0 % + 8×10 % = 5 %,
инвариантность к порядку строк и дублированию, масштабирование).
Используется в: ``run_experiment.py``, ``checks/``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .measure import WaveMeasurement

ABS_KEYS = ("A1", "t1", "A2", "t2")


def absolute_errors(etalon: WaveMeasurement,
                    estimate: WaveMeasurement) -> dict[str, float | None]:
    """Четыре абсолютные ошибки одной волны (§2).

    Значение ``None`` — пик не измерен на одной из сторон или второй пик
    не размечен у эталона (тогда ошибка не вычисляется).
    """
    out: dict[str, float | None] = {}
    for key in ABS_KEYS:
        ref = getattr(etalon, key)
        got = getattr(estimate, key)
        if key.startswith("A2") or key.startswith("t2"):
            if not etalon.second_required:
                out[key] = None
                continue
        out[key] = None if (ref is None or got is None) else abs(ref - got)
    return out


def normalized_errors(etalon: WaveMeasurement,
                      estimate: WaveMeasurement) -> dict[str, float | None]:
    """Нормированные ошибки одной волны в процентах (§2).

    Делители: ``A1`` эталона для амплитуд, ``T`` эталона для времён.
    Если делитель нулевой или пик не измерен, значение ``None``.
    """
    abs_err = absolute_errors(etalon, estimate)
    out: dict[str, float | None] = {}
    for key in ABS_KEYS:
        err = abs_err[key]
        if err is None:
            out[key] = None
            continue
        denom = etalon.A1 if key.startswith("A") else etalon.T
        if denom is None or denom == 0:
            out[key] = None
            continue
        out[key] = 100.0 * err / denom
    return out


def _flat_mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def nested_mean(groups: Mapping[Any, Mapping[Any, Sequence[float | None]]]
                ) -> float | None:
    """Среднее с равными весами эталонов (§2).

    ``groups[etalon][realization] -> последовательность ошибок по волнам``.
    Сначала среднее по волнам реализации, затем по реализациям, затем по
    эталонам. Пустые реализации и эталоны пропускаются; если значений нет
    вовсе, возвращается ``None`` (а не ноль).
    """
    per_etalon: list[float] = []
    for realizations in groups.values():
        per_realization = [_flat_mean(waves) for waves in realizations.values()]
        etalon_mean = _flat_mean(v for v in per_realization if v is not None)
        if etalon_mean is not None:
            per_etalon.append(etalon_mean)
    return _flat_mean(per_etalon)


def refusal_stats(expected: int, successes: int) -> dict[str, float | int | None]:
    """Число успешных и ожидаемых измерений и доля отказов в процентах.

    При нулевом числе ожидаемых измерений доля помечается ``None``
    (неприменима), а не нулём (§2).
    """
    if expected < 0 or successes < 0 or successes > expected:
        raise ValueError(f"некорректные счётчики: expected={expected}, "
                         f"successes={successes}")
    share = None if expected == 0 else 100.0 * (expected - successes) / expected
    return {"expected": expected, "successes": successes, "refusal_pct": share}
