"""Проверки фильтрации: усиление и фаза на длинных синусоидах (§5).

Ожидаемое усиление берётся из независимой формулы ``design/filter_reference.py``
(аналоговый прототип с предыскажением), а не из ``sosfiltfilt``. Допуски —
согласованные в §5: ``|G_изм - G_ожид| <= max(1e-5*G_ожид, 1e-9)``,
``|фаза| <= 0.01°`` в полосе (``G_ожид >= 1e-3``).

Данные: синусоиды порождаются формулами; файлы не читаются.
Проверка: ``verify.py``.
Используется в: ``verify.py`` (обязательные проверки).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfreqz

from ppg import butter_bandpass, moving_average
from reference import expected_gain, sine_gain_phase

FS = 320.0
DURATION = 300.0        # с, §5: длинные синусоиды
MEASURE_FROM = 60.0     # с, окно измерения вдали от краёв
MEASURE_TO = 240.0

CONFIGS = [
    (0.10, 3.0, 4, [0.10, 1.0, 3.0, 8.0]),
    (0.30, 5.0, 4, [0.30, 1.0, 5.0, 12.0]),
    (0.50, 8.0, 2, [0.50, 2.0, 8.0, 30.0]),
    (0.50, 20.0, 2, [0.50, 5.0, 20.0, 80.0]),
]


def _sine(freq: float) -> np.ndarray:
    t = np.arange(int(round(DURATION * FS)), dtype=np.float64) / FS
    return np.cos(2.0 * np.pi * freq * t)


def test_sine_gain_and_phase() -> None:
    """Два прохода, верные частоты и порядок: усиление и фаза совпадают."""
    for f_low, f_high, N, freqs in CONFIGS:
        for freq in freqs:
            y = butter_bandpass(_sine(freq), f_low, f_high, N, fs=FS)
            gain, phase_deg = sine_gain_phase(
                y, freq, MEASURE_FROM, MEASURE_TO, fs=FS)
            g_exp = expected_gain(freq, f_low, f_high, N, fs=FS)
            tol = max(1e-5 * g_exp, 1e-9)
            label = f"f={freq} Гц, полоса {f_low}–{f_high} Гц, N={N}"
            assert abs(gain - g_exp) <= tol, (
                f"{label}: усиление {gain:.12g}, ожидалось {g_exp:.12g}, "
                f"допуск {tol:.3g}")
            if g_exp >= 1e-3:
                assert abs(phase_deg) <= 0.01, (
                    f"{label}: фаза {phase_deg:.6g}°, допуск 0.01°")


def test_wrong_configurations_are_rejected() -> None:
    """Проверка чувствительна к одному проходу и к неверной частоте."""
    # на частоте среза однопроходное усиление |H| = 1/sqrt(2), а два прохода
    # дают |H|^2 = 0.5 — здесь ошибка «один проход» видна
    freq, f_low, f_high, N = 0.5, 0.5, 5.0, 4
    y = butter_bandpass(_sine(freq), f_low, f_high, N, fs=FS)
    gain, _ = sine_gain_phase(y, freq, MEASURE_FROM, MEASURE_TO, fs=FS)
    g_exp = expected_gain(freq, f_low, f_high, N, fs=FS)
    assert g_exp == pytest.approx(0.5, abs=1e-3)
    assert abs(gain - g_exp) <= max(1e-5 * g_exp, 1e-9)

    # один проход даёт |H|, а не |H|^2 — расхождение должно быть замечено
    from scipy.signal import sosfilt

    sos = butter(N, [f_low, f_high], btype="bandpass", fs=FS, output="sos")
    y1 = sosfilt(sos, _sine(freq))
    gain1, _ = sine_gain_phase(y1, freq, MEASURE_FROM, MEASURE_TO, fs=FS)
    assert abs(gain1 - g_exp) > 1e-2

    # неверная полоса (0.5 -> 2.0 Гц) при 4 Гц даёт другое усиление
    y2 = butter_bandpass(_sine(4.0), 0.5, 2.0, N, fs=FS)
    gain2, _ = sine_gain_phase(y2, 4.0, MEASURE_FROM, MEASURE_TO, fs=FS)
    assert abs(gain2 - expected_gain(4.0, 0.5, 5.0, N, fs=FS)) > 1e-3


def test_expected_gain_matches_sosfreqz() -> None:
    """Независимая формула АЧХ совпадает с ``scipy.signal.sosfreqz``."""
    grid = np.concatenate([
        np.array([0.10, 0.30, 0.50, 1.0, 2.0, 3.0, 5.0]),
        np.geomspace(0.1, 100.0, 40),
    ])
    worst = 0.0
    for f_low, f_high, N, _ in CONFIGS:
        sos = butter(N, [f_low, f_high], btype="bandpass", fs=FS, output="sos")
        _, h = sosfreqz(sos, worN=grid, fs=FS)
        ref = np.array([expected_gain(f, f_low, f_high, N, fs=FS) for f in grid])
        rel = np.abs(np.abs(h) ** 2 - ref) / np.maximum(ref, 1e-300)
        rel = rel[ref > 1e-12]
        worst = max(worst, float(rel.max()))
    assert worst < 1e-8, f"максимальное относительное расхождение {worst:.3e}"


@pytest.mark.parametrize("window", [5, 11, 21, 41])
def test_moving_average_is_unbiased_and_odd(window: int) -> None:
    """Скользящее среднее сохраняет константу и не сдвигает импульс."""
    x = np.full(1000, 2.5)
    assert np.allclose(moving_average(x, window), 2.5, atol=1e-12, rtol=0)

    impulse = np.zeros(1000)
    impulse[500] = 1.0
    y = moving_average(impulse, window)
    bottom = np.flatnonzero(y)
    assert bottom.min() == 500 - (window - 1) // 2
    assert bottom.max() == 500 + (window - 1) // 2
    assert y[500] == pytest.approx(1.0 / window, abs=1e-12)

    with pytest.raises(ValueError):
        moving_average(x, 4)
