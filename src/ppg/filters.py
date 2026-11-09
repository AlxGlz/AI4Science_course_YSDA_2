"""Фильтрация по §4 методики: полосовой Баттерворт и скользящее среднее.

Параметры и реализация зафиксированы методикой:

* ``butter(N, [f_low, f_high], btype='bandpass', fs=320, output='sos')``;
* ``sosfiltfilt(sos, signal, padtype='odd', padlen=None)`` — вперёд и назад;
* ``uniform_filter1d(signal, size=window, mode='reflect', origin=0)``.

Все вычисления в ``float64``; обработка всей записи до выделения волн.

Данные: массивы сигналов; файлы не читаются.
Проверка: ``checks/test_filters.py`` (усиление и фаза на длинных синусоидах,
усиление против независимой формулы АЧХ, вклад края).
Используется в: ``run_experiment.py``, ``checks/``, ``plot_comparison.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

FS_DEFAULT = 320.0


def butter_bandpass(signal: np.ndarray, f_low: float, f_high: float, N: int,
                    fs: float = FS_DEFAULT) -> np.ndarray:
    """Полосовой Баттерворт с обработкой вперёд и назад (§4).

    ``N`` — параметр ``scipy.signal.butter``; порядок полосового фильтра за
    один проход равен ``2*N``. Частоты в Гц, ``0 < f_low < f_high < fs/2``.
    """
    if not 0.0 < f_low < f_high < fs / 2.0:
        raise ValueError(f"некорректные частоты среза: {f_low}, {f_high} при fs={fs}")
    if int(N) != N or N < 1:
        raise ValueError(f"некорректный порядок N={N}")
    sos = butter(int(N), [f_low, f_high], btype="bandpass", fs=fs, output="sos")
    x = np.asarray(signal, dtype=np.float64)
    return sosfiltfilt(sos, x, padtype="odd", padlen=None)


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Центрированное скользящее среднее с равными весами (§2, §4).

    Окно должно быть нечётным (5, 11, 21, 41 отсчёт по §2).
    """
    if int(window) != window or window < 1 or window % 2 == 0:
        raise ValueError(f"окно должно быть нечётным положительным: {window}")
    x = np.asarray(signal, dtype=np.float64)
    return uniform_filter1d(x, size=int(window), mode="reflect", origin=0)
