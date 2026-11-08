"""Исследование краевых эффектов: достаточен ли отступ 5 с от краёв записи.

Методика (§2) оценивает полные волны между 5-й и 55-й секундами 60-секундной
записи, то есть рассчитывает на отступ 5 с от каждого края. Здесь измеряется,
насколько сильно переходный процесс ``sosfiltfilt(sos, x, padtype='odd',
padlen=None)`` искажает синусоиду на расстоянии ``t`` от начала записи для всех
сочетаний частот среза и ``N`` из §2, включая нижнюю частоту 0,10 Гц.

Данные: нет, используются синтетические синусоиды.
Проверка: результаты пишутся в ``results/design/edge_margin.csv``;
усиление в установившемся режиме сверяется с независимой формулой
``design/filter_reference.py``.
Используется в: ``design/run_design.py`` (сводка допусков §5).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

import filter_reference as fr

FS = 320.0
F_LOW_GRID = (0.10, 0.30, 0.50)
F_HIGH_GRID = (3.0, 5.0, 8.0, 12.0, 20.0)
N_GRID = (2, 4)


def bandpass_sos(f_low: float, f_high: float, N: int, fs: float = FS) -> np.ndarray:
    """Полосовой фильтр Баттерворта точно так, как задано в §4 методики."""
    return butter(N, [f_low, f_high], btype="bandpass", fs=fs, output="sos")


def filter_like_experiment(x: np.ndarray, f_low: float, f_high: float, N: int,
                           fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Обработка вперёд и назад с параметрами §4 методики."""
    sos = bandpass_sos(f_low, f_high, N, fs)
    return sosfiltfilt(sos, x, padtype="odd", padlen=None), sos


def default_padlen(sos: np.ndarray) -> int:
    """Значение ``padlen``, которое ``sosfiltfilt`` выбирает по умолчанию."""
    n_sections = sos.shape[0]
    n_zeros = int(np.count_nonzero(sos[:, 2] == 0.0))
    n_poles = int(np.count_nonzero(sos[:, 5] == 0.0))
    return 3 * (2 * n_sections + 1 - min(n_zeros, n_poles))


def transient_profile(f_test: float, f_low: float, f_high: float, N: int,
                      duration: float = 240.0, fs: float = FS) -> dict:
    """Профиль отклонения отклика от установившейся синусоиды у начала записи.

    Возвращает время ``t`` (с), нормированное отклонение ``rel`` (доля
    установившейся амплитуды выхода) и параметры фильтра.
    """
    n = int(round(duration * fs))
    t = np.arange(n, dtype=np.float64) / fs
    x = np.sin(2.0 * np.pi * f_test * t)
    y, sos = filter_like_experiment(x, f_low, f_high, N, fs)

    # установившееся усиление берём из независимой формулы, а не из отклика
    gain = fr.expected_gain(f_test, f_low, f_high, N, fs)
    reference = gain * x
    rel = np.abs(y - reference) / gain

    return {"t": t, "rel": rel, "sos": sos, "gain": gain,
            "padlen": default_padlen(sos)}


def settling_time(t: np.ndarray, rel: np.ndarray, tol: float) -> float:
    """Первый момент, после которого отклонение уже не превышает ``tol``."""
    bad = np.flatnonzero(rel > tol)
    if bad.size == 0:
        return 0.0
    idx = bad[-1] + 1
    if idx >= t.size:
        return float("inf")
    return float(t[idx])


def deviation_at(t: np.ndarray, rel: np.ndarray, t_probe: float) -> float:
    """Нормированное отклонение в момент ``t_probe``."""
    i = int(np.argmin(np.abs(t - t_probe)))
    return float(rel[i])
