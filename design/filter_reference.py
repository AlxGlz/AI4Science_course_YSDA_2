"""Независимый расчёт ожидаемой частотной характеристики полосового фильтра.

Ожидаемое усиление и фаза получаются из параметров методики (частоты среза,
параметр ``N``, частота дискретизации), а не из проверяемой функции проекта:
аналоговый прототип Баттерворта порядка ``N`` с билинейным преобразованием
и предыскажением на частоте дискретизации. Прямое произведение
``sosfiltfilt`` даёт квадрат модуля однопроходной характеристики и нулевой фазовый
сдвиг, поэтому ожидаемое усиление — ``|H|**2``, ожидаемая фаза — 0.

Данные: нет, только параметры §2 и §4 методики.
Проверка: ``design/run_design.py`` сравнивает формулу с ``scipy.signal.freqz``
на сетке частот и печатает максимальное относительное расхождение.
Используется в: ``design/run_design.py``, ``design/edge_margin.py``; далее — ``checks/``.
"""

from __future__ import annotations

import numpy as np

FS = 320.0


def prewarp(f: float, fs: float = FS) -> float:
    """Аналоговая частота после предыскажения билинейного преобразования."""
    return 2.0 * fs * np.tan(np.pi * np.asarray(f, dtype=np.float64) / fs)


def single_pass_gain(f: float, f_low: float, f_high: float, N: int,
                     fs: float = FS) -> float:
    """Модуль однопроходной АЧХ полосового Баттерворта, порядок полосы ``2N``.

    |H(jW)|**2 = 1 / (1 + ((W**2 - W0**2) / (B * W))**(2N)),
    где W — предыскажённая частота, W0**2 = Wl * Wu, B = Wu - Wl.
    """
    w = prewarp(f, fs)
    wl = prewarp(f_low, fs)
    wu = prewarp(f_high, fs)
    w0sq = wl * wu
    bw = wu - wl
    ratio = (w * w - w0sq) / (bw * w)
    return float(1.0 / np.sqrt(1.0 + np.abs(ratio) ** (2 * N)))


def expected_gain(f: float, f_low: float, f_high: float, N: int,
                  fs: float = FS) -> float:
    """Ожидаемое усиление после ``sosfiltfilt``: квадрат модуля однопрохода."""
    return single_pass_gain(f, f_low, f_high, N, fs) ** 2


def expected_phase_rad() -> float:
    """Ожидаемый фазовый сдвиг ``sosfiltfilt`` — ноль (обработка вперёд и назад)."""
    return 0.0
