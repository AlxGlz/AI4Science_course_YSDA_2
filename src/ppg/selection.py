"""Правило подбора параметров по §7 методики.

Допустимость варианта: измеритель не отказывает ни на одном пике, а средние
нормированные ошибки первого пика не превышают 5 % (``E_A1`` и ``E_t1``).
Ранжирование допустимых вариантов: минимум ``max(E_A2, E_t2)`` после
усреднения §2.

При точном равенстве оценок:

* для скользящего среднего выбирается меньшее окно;
* для Баттерворта — меньший порядок ``N``, затем меньшая нижняя, затем
  меньшая верхняя частота.

Модуль не читает файлы и не применяет фильтры: на вход подаются сводки,
посчитанные вызывающим кодом (``mean_normalized_pct``, ``refusals``,
``settings``).

Данные: сводки ошибок и отказов по вариантам обработки.
Проверка: ``checks/test_selection.py`` (допустимость, оценка, равенство
оценок, отсутствие допустимых вариантов, чужие настройки).
Используется в: ``run_experiment.py`` (этап ``select``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

FIRST_PEAK_LIMIT_PCT = 5.0
SECOND_PEAK_KEYS = ("A2", "t2")


def peak_score(summary: Mapping[str, Any]) -> float | None:
    """``max(E_A2, E_t2)`` после усреднения или ``None``, если их нет (§7).

    Отсутствие второго пика в разметке даёт ``None`` по обеим ошибкам:
    оценить вариант по второму пику нельзя, и ноль вместо ``None`` был бы
    подменой результата.
    """
    values = summary.get("mean_normalized_pct") or {}
    present = [values.get(key) for key in SECOND_PEAK_KEYS]
    known = [float(value) for value in present if value is not None]
    return max(known) if known else None


def is_admissible(summary: Mapping[str, Any],
                  limit_pct: float = FIRST_PEAK_LIMIT_PCT) -> bool:
    """Допустим ли вариант по §7: без отказов и первый пик в пределах."""
    refusals = summary.get("refusals") or {}
    for peak in ("first_peak", "second_peak"):
        item = refusals.get(peak) or {}
        expected = item.get("expected")
        successes = item.get("successes")
        if expected is None or successes is None or successes != expected:
            return False
    values = summary.get("mean_normalized_pct") or {}
    a1 = values.get("A1")
    t1 = values.get("t1")
    if a1 is None or t1 is None:
        return False
    return float(a1) <= limit_pct and float(t1) <= limit_pct


def _settings(summary: Mapping[str, Any], kind: str) -> dict[str, Any]:
    settings = dict(summary.get("settings") or {})
    if settings.get("kind") != kind:
        raise ValueError(
            f"ожидались настройки {kind!r}, получено {settings.get('kind')!r}")
    return settings


def _best(candidates: Sequence[Mapping[str, Any]], kind: str,
          tie_break) -> Mapping[str, Any] | None:
    best: Mapping[str, Any] | None = None
    best_key: tuple | None = None
    for summary in candidates:
        settings = _settings(summary, kind)
        if not is_admissible(summary):
            continue
        score = peak_score(summary)
        if score is None:
            continue
        key = (score, *tie_break(settings))
        if best_key is None or key < best_key:
            best, best_key = summary, key
    return best


def choose_moving_average(candidates: Sequence[Mapping[str, Any]]
                          ) -> Mapping[str, Any] | None:
    """Допустимое среднее с минимальной оценкой; равенство — меньшее окно."""
    return _best(candidates, "moving_average",
                 lambda settings: (int(settings["window"]),))


def choose_butterworth(candidates: Sequence[Mapping[str, Any]]
                       ) -> Mapping[str, Any] | None:
    """Допустимый Баттерворт по §7: оценка, затем N, f_low, f_high."""
    def tie_break(settings: Mapping[str, Any]) -> tuple:
        return (int(settings["N"]), float(settings["f_low_hz"]),
                float(settings["f_high_hz"]))

    return _best(candidates, "butterworth", tie_break)


def count_admissible(candidates: Sequence[Mapping[str, Any]]) -> int:
    """Сколько вариантов проходят пороги §7 — для диагностики."""
    return sum(1 for summary in candidates if is_admissible(summary))
