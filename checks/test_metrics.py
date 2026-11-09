"""Проверки усреднения ошибок и учёта отказов по §2.

Данные: синтетические значения ошибок; файлы не читаются.
Проверка: ``verify.py``.
Используется в: ``verify.py`` (обязательные проверки).
"""

from __future__ import annotations

import random

import pytest

from ppg.metrics import nested_mean, refusal_stats

TOL = 1e-9          # допуск §5 для усреднения, п. п.


def _groups(waves_per_etalon: dict[str, list[float]],
            repeats: int = 5) -> dict[str, dict[str, list[float]]]:
    return {etalon: {f"rep{r}": list(waves) for r in range(repeats)}
            for etalon, waves in waves_per_etalon.items()}


def test_averaging_equal_weights() -> None:
    """2 волны × 0 % + 8 волн × 10 % = 5 %, а не 8 % (§5)."""
    groups = _groups({"E1": [0.0, 0.0], "E2": [10.0] * 8})
    result = nested_mean(groups)
    assert result == pytest.approx(5.0, abs=TOL)
    # плоское среднее всех волн дало бы 8 % — проверка различает схемы
    flat = sum(v for waves in groups.values()
               for rep in waves.values() for v in rep) / 50.0
    assert flat == pytest.approx(8.0, abs=TOL)
    assert abs(result - flat) > 1.0


def test_averaging_order_and_duplication() -> None:
    """Перестановка строк и одинаковое дублирование волн не меняют результат."""
    base = _groups({"E1": [0.0, 2.0], "E2": [10.0] * 8, "E3": [3.5]})
    reference = nested_mean(base)

    rng = random.Random(20261108)
    shuffled: dict[str, dict[str, list[float]]] = {}
    for etalon, reps in base.items():
        shuffled[etalon] = {}
        for rep, waves in reps.items():
            copy = list(waves)
            rng.shuffle(copy)
            shuffled[etalon][rep] = copy
    assert nested_mean(shuffled) == pytest.approx(reference, abs=TOL)

    duplicated = {etalon: {rep: [v for v in waves for _ in range(2)]
                           for rep, waves in reps.items()}
                  for etalon, reps in base.items()}
    assert nested_mean(duplicated) == pytest.approx(reference, abs=TOL)


def test_empty_groups_are_not_zero() -> None:
    """Пустая группа пропускается, а не заменяется нулём (§2)."""
    groups = {"E1": {"r0": []}, "E2": {"r0": [4.0]}}
    assert nested_mean(groups) == pytest.approx(4.0, abs=TOL)
    assert nested_mean({"E1": {"r0": []}}) is None
    assert nested_mean({}) is None


def test_refusal_stats() -> None:
    """Доля отказов в процентах; при нуле ожидаемых — неприменима."""
    stats = refusal_stats(expected=10, successes=7)
    assert stats["expected"] == 10
    assert stats["successes"] == 7
    assert stats["refusal_pct"] == pytest.approx(30.0, abs=TOL)

    empty = refusal_stats(expected=0, successes=0)
    assert empty["refusal_pct"] is None

    with pytest.raises(ValueError):
        refusal_stats(expected=3, successes=4)
