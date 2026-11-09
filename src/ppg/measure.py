"""Измерение пиков волны по правилам §2 методики.

Правила (METHOD.md, §2 «Измерение амплитуды и времени пиков»):

* поиск локальных максимумов на отрезке ``x[s:b+1]`` через
  ``scipy.signal.find_peaks`` без порогов и без дополнительного сглаживания;
* первый пик — самый высокий максимум с фазой ``0.10 <= (p-s)/L <= 0.40``,
  при равной высоте — более ранний, плоская вершина — средний отсчёт с
  округлением вниз (поведение ``find_peaks``);
* второй пик (только при ``has_diastolic_peak``) — максимум с наибольшей
  prominence среди кандидатов ``0.40 <= (p-s)/L <= 0.85`` строго после первого,
  при равной prominence — более ранний;
* амплитуды отсчитываются от уровня начала волны ``x[s]`` того же сигнала;
  времена — от начала волны; интерполяции нет, отказ фиксируется явно.

Данные: массивы сигналов (float64) и индексы разметки; файлы не читаются.
Проверка: ``checks/test_measure.py`` (CTRL-W1…W6, сдвиг уровня, масштаб,
известное смещение пика, демонстрация намеренной ошибки 320 → 250).
Используется в: ``src/ppg/__init__.py``, ``checks/``, ``run_experiment.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

FS_DEFAULT = 320.0          # Гц, §2 и §3 методики
_FIRST_LO, _FIRST_HI = 0.10, 0.40
_SECOND_LO, _SECOND_HI = 0.40, 0.85


@dataclass(frozen=True)
class WaveMeasurement:
    """Результат измерения одной волны.

    Поля ``None`` означают отказ измерения (или неприменимость второго пика,
    если ``second_required`` равно ``False``). Индексы ``p1``, ``p2`` — во всей
    записи; ``A1``, ``A2`` — условные единицы от ``x[s]``; ``t1``, ``t2``, ``T`` —
    миллисекунды.
    """

    p1: int | None
    p2: int | None
    A1: float | None
    A2: float | None
    t1: float | None
    t2: float | None
    T: float
    refused_first: bool
    refused_second: bool
    second_required: bool


def _local_maxima(x: np.ndarray, s: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Локальные максимумы отрезка ``x[s:b+1]`` и их prominence.

    Возвращает индексы во всей записи и prominence, посчитанную на полной
    волне без ограничения ``wlen`` (§2 п.3). Края ``s`` и ``b`` локальными
    максимумами не считаются — так же, как в ``find_peaks``.
    """
    segment = np.asarray(x, dtype=np.float64)[s:b + 1]
    idx, props = find_peaks(segment, prominence=(None, None))
    return idx + s, np.asarray(props["prominences"], dtype=np.float64)


def measure_wave(x: np.ndarray, s: int, b: int, has_diastolic_peak: bool,
                 fs: float = FS_DEFAULT) -> WaveMeasurement:
    """Измерить пики одной волны по §2.

    ``x`` — вся запись, ``s`` — ``start_idx``, ``b`` — ``end_idx`` разметки;
    длина волны ``L = b - s``. Уровень начала волны берётся из ``x[s]``.
    """
    if b <= s:
        raise ValueError(f"пустая волна: s={s}, b={b}")
    if s < 0 or b >= len(x):
        raise ValueError(f"индексы разметки вне записи: s={s}, b={b}, len={len(x)}")

    L = b - s
    peaks, prominences = _local_maxima(x, s, b)

    T = 1000.0 * L / fs
    phases = (peaks - s) / L

    def make(p1: int | None, p2: int | None) -> WaveMeasurement:
        return WaveMeasurement(
            p1=p1,
            p2=p2,
            A1=None if p1 is None else float(x[p1] - x[s]),
            A2=None if p2 is None else float(x[p2] - x[s]),
            t1=None if p1 is None else 1000.0 * (p1 - s) / fs,
            t2=None if p2 is None else 1000.0 * (p2 - s) / fs,
            T=T,
            refused_first=p1 is None,
            refused_second=has_diastolic_peak and p2 is None,
            second_required=has_diastolic_peak,
        )

    first_mask = (phases >= _FIRST_LO) & (phases <= _FIRST_HI)
    if not np.any(first_mask):
        return make(None, None)
    first_idx = np.flatnonzero(first_mask)
    # max по (высота, -индекс): при равной высоте побеждает более ранний
    best = max(first_idx, key=lambda i: (float(x[peaks[i]]), -int(peaks[i])))
    p1 = int(peaks[best])

    if not has_diastolic_peak:
        return make(p1, None)

    second_mask = ((phases >= _SECOND_LO) & (phases <= _SECOND_HI)
                   & (peaks > p1))
    if not np.any(second_mask):
        return make(p1, None)
    second_idx = np.flatnonzero(second_mask)
    # max по (prominence, -индекс): при равной prominence — более ранний
    best2 = max(second_idx, key=lambda i: (float(prominences[i]), -int(peaks[i])))
    return make(p1, int(peaks[best2]))


def measure_waves(x: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                  has_diastolic_peak: np.ndarray | bool,
                  fs: float = FS_DEFAULT) -> list[WaveMeasurement]:
    """Измерить пики на последовательности волн одной записи."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if np.ndim(has_diastolic_peak) == 0:
        flags = [bool(has_diastolic_peak)] * len(starts)
    else:
        flags = [bool(v) for v in np.asarray(has_diastolic_peak)]
    if not (len(starts) == len(ends) == len(flags)):
        raise ValueError("длины starts, ends и флагов не совпадают")
    return [measure_wave(x, int(s), int(b), f, fs=fs)
            for s, b, f in zip(starts, ends, flags)]
