"""Проверки статистики §7.5: поправка Холма и парный односторонний тест.

Ожидаемые p-значения независимы от проверяемого кода: для известного
примера они посчитаны вручную и сверяются с ``scipy.stats.t`` напрямую.
Данные: нет.
Проверка: сам модуль ``src/ppg/stats.py``.
Используется в: ``verify.py`` через ``checks/``.
"""

from __future__ import annotations

import math

import pytest
from scipy import stats as scipy_stats

from ppg.stats import family_p_values, holm_adjust, paired_test


def test_holm_known_example_preserves_input_order():
    # p = (0,01; 0,04; 0,03; 0,005); порядок по возрастанию 0,005 < 0,01
    # < 0,03 < 0,04 даёт множители 4, 3, 2, 1.
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_is_monotone_and_capped_at_one():
    adjusted = holm_adjust([0.001, 0.02, 0.04, 0.9])
    assert adjusted == sorted(adjusted)
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    # Равные p-значения дают 4·0,5 = 2 и ограничиваются единицей.
    assert holm_adjust([0.5, 0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0, 1.0]


def test_holm_of_empty_family_is_empty():
    assert holm_adjust([]) == []


def test_holm_rejects_invalid_p_values():
    with pytest.raises(ValueError):
        holm_adjust([0.01, 1.2])
    with pytest.raises(ValueError):
        holm_adjust([math.nan])


def test_paired_test_matches_scipy_one_sided():
    baseline = [3.0, 4.0, 5.0, 6.0, 7.5, 5.5]
    candidate = [1.0, 2.5, 2.0, 3.5, 3.0, 4.0]
    result = paired_test(baseline, candidate, alternative="greater")
    reference = scipy_stats.ttest_rel(baseline, candidate,
                                      alternative="greater")
    assert result["n"] == 6
    assert result["mean_difference"] == pytest.approx(2.5)
    assert result["statistic"] == pytest.approx(float(reference.statistic))
    assert result["p_value"] == pytest.approx(float(reference.pvalue))


def test_paired_test_known_values():
    # Разности 2, 2, 3, 3: среднее 2,5, выборочное СКО sqrt(1/3),
    # t = 2,5 / (0,57735 / 2) = 8,6603 при 3 степенях свободы.
    result = paired_test([3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 2.0, 3.0])
    assert result["mean_difference"] == pytest.approx(2.5)
    assert result["standard_deviation"] == pytest.approx(math.sqrt(1.0 / 3.0))
    assert result["statistic"] == pytest.approx(8.660254, rel=1e-6)
    expected = float(scipy_stats.t.sf(8.660254, df=3))
    assert result["p_value"] == pytest.approx(expected)


def test_paired_test_degenerate_differences():
    positive = paired_test([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    assert positive["standard_deviation"] == 0.0
    assert positive["p_value"] == 0.0
    assert positive["statistic"] == math.inf

    negative = paired_test([1.0, 2.0], [2.0, 3.0])
    assert negative["p_value"] == 1.0

    equal = paired_test([1.0, 2.0], [1.0, 2.0])
    assert equal["statistic"] == 0.0
    assert equal["p_value"] == 1.0


def test_paired_test_needs_two_pairs():
    result = paired_test([1.0], [0.5])
    assert result["n"] == 1
    assert result["p_value"] is None
    assert result["statistic"] is None


def test_paired_test_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_test([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_test([1.0, math.nan], [1.0, 2.0])
    with pytest.raises(ValueError):
        paired_test([1.0, 2.0], [1.0, 2.0], alternative="both")


def test_family_p_values_substitutes_one_for_missing():
    comparisons = [
        {"p_value": 0.02},
        {"p_value": None},
        {"p_value": 0.04},
        {"p_value": 0.001},
    ]
    assert family_p_values(comparisons) == [0.02, 1.0, 0.04, 0.001]
    # Порядок: 0,001 (×4), 0,02 (×3), 0,04 (×2), 1 (×1).
    adjusted = holm_adjust(family_p_values(comparisons))
    assert adjusted == pytest.approx([0.06, 1.0, 0.08, 0.004])
