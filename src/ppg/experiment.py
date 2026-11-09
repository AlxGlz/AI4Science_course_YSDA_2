"""Оцениваемые волны и ошибки обработки относительно чистого эталона (§2).

Правила (METHOD.md, §2):

* оцениваются только полные волны, целиком лежащие между 5-й и 55-й
  секундами записи: ``start_idx >= 5 * fs`` и ``end_idx <= 55 * fs``;
* один и тот же измеритель применяется к эталону и к каждой обработке;
* для каждой волны считаются четыре абсолютные ошибки и четыре
  нормированные ошибки в процентах (нормировка — до усреднения);
* отказ измерителя не заменяется нулевой ошибкой: число успешных и
  ожидаемых измерений выводится отдельно по каждому пику.

Модуль не читает файлы и не применяет фильтры: сигналы и разметка
передаются вызывающей стороной, что отделяет вычисления от ввода-вывода.

Данные: массивы сигналов, ``ppg.io.WaveTable`` разметки.
Проверка: ``checks/test_experiment.py`` (границы окна 5–55 с, точные ошибки
на сдвинутом сигнале, подсчёт отказов и неприменимый второй пик).
Используется в: ``run_experiment.py`` (этапы ``test``, ``select``, ``main``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import WaveTable
from .measure import FS_DEFAULT, WaveMeasurement, measure_waves
from .metrics import ABS_KEYS, absolute_errors, normalized_errors, refusal_stats

EVAL_START_SEC = 5.0
EVAL_END_SEC = 55.0


def select_evaluation_mask(waves: WaveTable,
                           fs: float = FS_DEFAULT) -> np.ndarray:
    """Маска волн, целиком лежащих между 5-й и 55-й секундами (§2).

    Волна попадает в оценку, если её начало не раньше 5 с и конец не позже
    55 с. При ``fs = 320`` Гц границы равны 1600 и 17600 отсчётам.
    """
    if not fs > 0:
        raise ValueError(f"частота дискретизации должна быть положительной: {fs}")
    starts = np.asarray(waves.start_idx, dtype=np.int64)
    ends = np.asarray(waves.end_idx, dtype=np.int64)
    if starts.shape != ends.shape:
        raise ValueError("длины start_idx и end_idx не совпадают")
    start_limit = EVAL_START_SEC * fs
    end_limit = EVAL_END_SEC * fs
    return (starts >= start_limit) & (ends <= end_limit)


@dataclass(frozen=True)
class WaveComparison:
    """Измерения эталона и обработки на одной оцениваемой волне."""

    wave_index: int
    start_idx: int
    end_idx: int
    has_diastolic_peak: bool
    etalon: WaveMeasurement
    estimate: WaveMeasurement
    absolute: dict[str, float | None]
    normalized: dict[str, float | None]


def compare_waves(waves: WaveTable, etalon_signal: np.ndarray,
                  estimate_signal: np.ndarray,
                  mask: np.ndarray | None = None,
                  fs: float = FS_DEFAULT) -> list[WaveComparison]:
    """Измерить эталон и обработку и посчитать ошибки по волнам (§2).

    ``mask`` — результат :func:`select_evaluation_mask`; ``None`` означает
    все волны разметки. Порядок волн сохраняется.
    """
    etalon = np.asarray(etalon_signal, dtype=np.float64)
    estimate = np.asarray(estimate_signal, dtype=np.float64)
    if etalon.ndim != 1 or estimate.ndim != 1:
        raise ValueError("сигналы должны быть одномерными")
    if etalon.shape != estimate.shape:
        raise ValueError(
            f"длины сигналов различаются: {etalon.size} и {estimate.size}")
    if not np.all(np.isfinite(etalon)):
        raise ValueError("в эталонном сигнале есть NaN или бесконечность")
    if not np.all(np.isfinite(estimate)):
        raise ValueError("в обработанном сигнале есть NaN или бесконечность")

    if mask is None:
        indices = np.arange(len(waves), dtype=np.int64)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(waves),):
            raise ValueError("маска не совпадает по длине с разметкой")
        indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("в окне оценки 5–55 с нет полных волн")

    starts = np.asarray(waves.start_idx, dtype=np.int64)[indices]
    ends = np.asarray(waves.end_idx, dtype=np.int64)[indices]
    flags = np.asarray(waves.has_diastolic_peak)[indices]

    etalon_measurements = measure_waves(etalon, starts, ends, flags, fs=fs)
    estimate_measurements = measure_waves(estimate, starts, ends, flags, fs=fs)

    comparisons: list[WaveComparison] = []
    for position, wave_index in enumerate(indices):
        reference = etalon_measurements[position]
        measured = estimate_measurements[position]
        comparisons.append(WaveComparison(
            wave_index=int(wave_index),
            start_idx=int(starts[position]),
            end_idx=int(ends[position]),
            has_diastolic_peak=bool(flags[position]),
            etalon=reference,
            estimate=measured,
            absolute=absolute_errors(reference, measured),
            normalized=normalized_errors(reference, measured),
        ))
    return comparisons


def error_series(comparisons: list[WaveComparison],
                 normalized: bool = False) -> dict[str, list[float | None]]:
    """Четыре ряда ошибок по волнам: абсолютные или нормированные (§2)."""
    extract = (lambda item: item.normalized) if normalized else (
        lambda item: item.absolute)
    return {key: [extract(item)[key] for item in comparisons]
            for key in ABS_KEYS}


def value_counts(series: dict[str, list[float | None]]) -> dict[str, int]:
    """Число волн с вычисленной ошибкой — знаменатель среднего (§2)."""
    return {key: sum(1 for value in values if value is not None)
            for key, values in series.items()}


def refusal_summary(comparisons: list[WaveComparison]) -> dict[str, dict]:
    """Отказы измерителя на обработке по каждому пику (§2)."""
    expected_first = len(comparisons)
    success_first = sum(1 for item in comparisons
                        if item.estimate.p1 is not None)
    flagged = [item for item in comparisons if item.has_diastolic_peak]
    success_second = sum(1 for item in flagged if item.estimate.p2 is not None)
    return {
        "first_peak": refusal_stats(expected_first, success_first),
        "second_peak": refusal_stats(len(flagged), success_second),
    }


def etalon_refusal_summary(comparisons: list[WaveComparison]) -> dict[str, dict]:
    """Отказы измерителя на чистом эталоне — контроль пригодности данных."""
    expected_first = len(comparisons)
    success_first = sum(1 for item in comparisons
                        if item.etalon.p1 is not None)
    flagged = [item for item in comparisons if item.etalon.second_required]
    success_second = sum(1 for item in flagged if item.etalon.p2 is not None)
    return {
        "first_peak": refusal_stats(expected_first, success_first),
        "second_peak": refusal_stats(len(flagged), success_second),
    }
