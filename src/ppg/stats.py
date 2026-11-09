"""Парное сравнение ошибок и поправка Холма (§7.5).

Единица наблюдения — эталон после усреднения ошибок по волнам и пяти
шумовым реализациям (§2, §7.5). Для каждой пары считается разность
``база − обработка``: положительная разность означает, что обработка
ближе к эталону, чем база. Тест односторонний: проверяется гипотеза
«средняя разность больше нуля».

Поправка Холма применяется совместно ко всем p-значениям семейства
сравнений (§7.5). Для невыполненного сравнения в поправку подставляется
``1``, а в таблицах оно помечается как «не выполнено»; порог ``0,05``
не ослабляется (§7.7).

Вырожденный случай — все парные разности равны. Тогда выборочное
стандартное отклонение нулевое, и t-статистика не определена; принимаем
соглашение: при нулевой средней разности ``p = 1``, при положительной
``p = 0``, при отрицательной ``p = 1``. Это решение фиксируется в
результатах, а не подбирается по данным.

Данные: последовательности парных значений (проценты, условные единицы,
миллисекунды); файлы не читаются.
Проверка: ``checks/test_stats.py`` (поправка Холма на известном примере,
монотонность и ограничение единицей, совпадение с ``scipy.stats.ttest_rel``,
вырожденные пары, отказ при неполных данных).
Используется в: ``run_experiment.py`` (этап ``main``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy import stats as scipy_stats

ALPHA_DEFAULT = 0.05
MISSING_P_VALUE = 1.0
ALTERNATIVES = ("greater", "less", "two-sided")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Поправка Холма с сохранением порядка входных p-значений (§7.5).

    Значения упорядочиваются по возрастанию, умножаются на число оставшихся
    гипотез и ограничиваются сверху единицей; монотонность восстанавливается
    максимумом по уже обработанным. Пустое семейство даёт пустой список.
    """
    values = [float(value) for value in p_values]
    for value in values:
        if math.isnan(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"p-значение вне [0, 1]: {value!r}")
    count = len(values)
    order = sorted(range(count), key=lambda index: values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_test(baseline: Sequence[float], candidate: Sequence[float],
                alternative: str = "greater") -> dict[str, object]:
    """Односторонний парный t-тест для разностей ``база − обработка`` (§7.5).

    Пары должны быть полными: одна запись на эталон, без пропусков.
    ``alternative='greater'`` проверяет, что средняя разность положительна,
    то есть обработка точнее базы. Меньше двух пар — сравнение невыполнимо,
    возвращается ``n`` с пустыми ``statistic`` и ``p_value``.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"неизвестная альтернатива: {alternative!r}")
    left = [float(value) for value in baseline]
    right = [float(value) for value in candidate]
    if len(left) != len(right):
        raise ValueError(
            f"длины парных рядов различаются: {len(left)} и {len(right)}")
    for value in left + right:
        if not math.isfinite(value):
            raise ValueError("в парных данных есть NaN или бесконечность")

    count = len(left)
    result: dict[str, object] = {
        "n": count,
        "mean_difference": None,
        "standard_deviation": None,
        "statistic": None,
        "p_value": None,
        "alternative": alternative,
    }
    if count < 2:
        return result

    differences = [a - b for a, b in zip(left, right)]
    mean = sum(differences) / count
    variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
    deviation = math.sqrt(variance)
    result["mean_difference"] = mean
    result["standard_deviation"] = deviation

    if deviation == 0.0:
        if mean > 0.0:
            result["statistic"] = math.inf
            result["p_value"] = 0.0
        elif mean < 0.0:
            result["statistic"] = -math.inf
            result["p_value"] = 1.0
        else:
            result["statistic"] = 0.0
            result["p_value"] = 1.0
        return result

    statistic = mean / (deviation / math.sqrt(count))
    distribution = scipy_stats.t(df=count - 1)
    if alternative == "greater":
        p_value = float(distribution.sf(statistic))
    elif alternative == "less":
        p_value = float(distribution.cdf(statistic))
    else:
        p_value = float(2.0 * min(distribution.cdf(statistic),
                                  distribution.sf(statistic)))
    result["statistic"] = statistic
    result["p_value"] = min(1.0, max(0.0, p_value))
    return result


def family_p_values(comparisons: Sequence[dict[str, object]]) -> list[float]:
    """p-значения семейства сравнений; невыполненное даёт ``1`` (§7.5)."""
    values: list[float] = []
    for item in comparisons:
        p_value = item.get("p_value")
        values.append(MISSING_P_VALUE if p_value is None else float(p_value))
    return values
