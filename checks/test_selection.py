"""Проверки правила подбора §7: допустимость, оценка, равенство оценок.

Сводки порождаются прямо в проверках; файлы и фильтры не нужны.
Данные: нет.
Проверка: сам модуль ``src/ppg/selection.py``.
Используется в: ``verify.py`` через ``checks/``.
"""

from __future__ import annotations

import pytest

from ppg.selection import (
    choose_butterworth,
    choose_moving_average,
    count_admissible,
    is_admissible,
    peak_score,
)


def _summary(kind: str = "butterworth", *, a1: float | None = 1.0,
             t1: float | None = 1.0, a2: float | None = 2.0,
             t2: float | None = 3.0, refusals: tuple[int, int] = (0, 0),
             **settings) -> dict:
    first_refusals, second_refusals = refusals
    return {
        "settings": {"kind": kind, **settings},
        "mean_normalized_pct": {"A1": a1, "t1": t1, "A2": a2, "t2": t2},
        "refusals": {
            "first_peak": {"expected": 100,
                           "successes": 100 - first_refusals},
            "second_peak": {"expected": 50,
                            "successes": 50 - second_refusals},
        },
    }


def _butterworth(score: float, *, a2: float | None = None,
                 t2: float | None = None, refusals: tuple[int, int] = (0, 0),
                 **settings) -> dict:
    return _summary("butterworth", a2=a2 if a2 is not None else score,
                    t2=t2 if t2 is not None else score, refusals=refusals,
                    **settings)


def _moving_average(score: float, window: int, *,
                    refusals: tuple[int, int] = (0, 0)) -> dict:
    return _summary("moving_average", a2=score, t2=score,
                    refusals=refusals, window=window)


def test_peak_score_is_maximum_of_second_peak_errors():
    assert peak_score(_summary(a2=2.0, t2=3.5)) == 3.5


def test_peak_score_skips_missing_errors():
    assert peak_score(_summary(a2=None, t2=4.0)) == 4.0
    assert peak_score(_summary(a2=None, t2=None)) is None


def test_admissible_requires_no_refusals():
    assert is_admissible(_summary()) is True
    assert is_admissible(_summary(refusals=(1, 0))) is False
    assert is_admissible(_summary(refusals=(0, 2))) is False


def test_admissible_checks_first_peak_limits_inclusive():
    assert is_admissible(_summary(a1=5.0, t1=5.0)) is True
    assert is_admissible(_summary(a1=5.01)) is False
    assert is_admissible(_summary(t1=5.01)) is False
    assert is_admissible(_summary(a1=None)) is False


def test_admissible_respects_custom_limit():
    assert is_admissible(_summary(a1=3.0), limit_pct=2.0) is False


def test_moving_average_takes_minimum_score():
    candidates = [_moving_average(4.0, 5), _moving_average(3.0, 11),
                  _moving_average(5.0, 21)]
    chosen = choose_moving_average(candidates)
    assert chosen is not None and chosen["settings"]["window"] == 11


def test_moving_average_tie_prefers_smaller_window():
    candidates = [_moving_average(3.0, 21), _moving_average(3.0, 5)]
    chosen = choose_moving_average(candidates)
    assert chosen is not None and chosen["settings"]["window"] == 5


def test_butterworth_tie_prefers_order_then_lower_band():
    same_score = 1.0
    candidates = [
        _butterworth(same_score, f_low_hz=0.10, f_high_hz=3.0, N=4),
        _butterworth(same_score, f_low_hz=0.50, f_high_hz=20.0, N=2),
        _butterworth(same_score, f_low_hz=0.30, f_high_hz=8.0, N=2),
        _butterworth(same_score, f_low_hz=0.30, f_high_hz=3.0, N=2),
    ]
    chosen = choose_butterworth(candidates)
    assert chosen is not None
    settings = chosen["settings"]
    assert (settings["N"], settings["f_low_hz"], settings["f_high_hz"]) == (
        2, 0.30, 3.0)


def test_butterworth_skips_best_scoring_inadmissible_variant():
    candidates = [
        _butterworth(0.5, f_low_hz=0.30, f_high_hz=8.0, N=2,
                     refusals=(3, 0)),
        _butterworth(2.0, f_low_hz=0.10, f_high_hz=3.0, N=4),
    ]
    chosen = choose_butterworth(candidates)
    assert chosen is not None
    assert chosen["settings"]["N"] == 4


def test_no_admissible_variant_returns_none():
    candidates = [
        _summary("butterworth", a1=9.0, f_low_hz=0.30, f_high_hz=8.0, N=2),
        _summary("moving_average", a1=9.0, window=11),
    ]
    assert choose_butterworth(candidates[:1]) is None
    assert choose_moving_average(candidates[1:]) is None
    assert count_admissible(candidates) == 0


def test_count_admissible_counts_both_kinds():
    candidates = [_butterworth(1.0, f_low_hz=0.30, f_high_hz=8.0, N=2),
                  _butterworth(1.0, f_low_hz=0.50, f_high_hz=8.0, N=2,
                               refusals=(1, 0)),
                  _moving_average(2.0, 5)]
    assert count_admissible(candidates) == 2


def test_wrong_kind_of_settings_is_rejected():
    with pytest.raises(ValueError, match="ожидались настройки"):
        choose_moving_average([_summary("butterworth", f_low_hz=0.30,
                                        f_high_hz=8.0, N=2)])
