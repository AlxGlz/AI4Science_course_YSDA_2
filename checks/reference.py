"""Независимые источники ожидаемых ответов для верификатора.

Модуль связывает три независимых источника:

* ``design.controls`` — контрольные сигналы и ожидаемые ответы, полученные
  прямым перебором локальных максимумов (не ``find_peaks`` и не ``src/ppg``);
* ``design.filter_reference`` — ожидаемая АЧХ из параметров методики
  (аналоговый прототип с предыскажением), независимая от ``sosfiltfilt``;
* ``AGREED`` ниже — численные ответы, согласованные в §5 ``METHOD.md``
  (2026-11-08 23:35 +02:00). Это отдельная, «бумажная» фиксация: если
  генератор в ``design/`` случайно изменится, проверка расхождения заметит.

Данные: контрольные сигналы порождаются формулами §5; файлы не читаются.
Проверка: ``checks/test_measure.py::test_control_signals_match_method``
сверяет независимый перебор с ``AGREED``, ``checks/test_filters.py``
сверяет формулу с ``sosfreqz``.
Используется в: ``checks/test_measure.py``, ``checks/test_filters.py``.
"""

from __future__ import annotations

import numpy as np

from design import controls
from design.filter_reference import expected_gain, expected_phase_rad

FS = 320.0
WAVE_LEN = 320

# Согласованные значения §5 METHOD.md: (p1, A1, t1, p2, A2, t2).
# Значение None — отказ или неприменимость.
AGREED: dict[str, dict[str, float | int | None]] = {
    "CTRL-W1": {"p1": 64, "A1": 2.0, "t1": 200.0,
                "p2": 192, "A2": 1.0, "t2": 600.0},
    "CTRL-W2": {"p1": 64, "A1": 2.0, "t1": 200.0,
                "p2": None, "A2": None, "t2": None},
    "CTRL-W3": {"p1": 64, "A1": 2.0, "t1": 200.0,
                "p2": None, "A2": None, "t2": None},
    "CTRL-W4": {"p1": 63, "A1": 1.99, "t1": 196.875,
                "p2": None, "A2": None, "t2": None},
    "CTRL-W5": {"p1": 64, "A1": 2.0, "t1": 200.0,
                "p2": 240, "A2": 0.8, "t2": 750.0},
    "CTRL-W6a": {"p1": 68, "A1": 2.0, "t1": 212.5,
                 "p2": 192, "A2": 1.0, "t2": 600.0},
    "CTRL-W6b": {"p1": 60, "A1": 2.0, "t1": 187.5,
                 "p2": 192, "A2": 1.0, "t2": 600.0},
}

# Флаг разметки «второй пик есть» для каждого контрольного сигнала.
HAS_SECOND: dict[str, bool] = {
    "CTRL-W1": True,
    "CTRL-W2": True,      # размечен, но локального максимума в окне нет
    "CTRL-W3": False,     # класс без второго пика
    "CTRL-W4": True,      # размечен, в окне 0.40..0.85 максимума нет
    "CTRL-W5": True,
    "CTRL-W6a": True,
    "CTRL-W6b": True,
}


def control_signals() -> dict[str, np.ndarray]:
    """Построить все сигналы CTRL-W1…W6b по формулам §5."""
    return {
        "CTRL-W1": controls.two_peak_wave(),
        "CTRL-W2": controls.shoulder_wave(),
        "CTRL-W3": controls.single_peak_wave(),
        "CTRL-W4": controls.flat_top_wave(),
        "CTRL-W5": controls.prominence_wave(),
        "CTRL-W6a": controls.shifted_first_peak_wave(4),
        "CTRL-W6b": controls.shifted_first_peak_wave(-4),
    }


def independent_measurement(signal: np.ndarray, has_second: bool) -> dict:
    """Ожидаемое измерение прямым перебором (``design.controls``)."""
    return controls.expected_measurement(signal, has_second, 0, WAVE_LEN)


def sine_gain_phase(signal: np.ndarray, freq: float, t_start: float,
                    t_end: float, fs: float = FS) -> tuple[float, float]:
    """Усиление и фазовый сдвиг синусоиды по демодуляции на окне.

    Частоты выбираются так, чтобы в окне укладывалось целое число периодов:
    тогда среднее произведений точное, без утечки. Входная амплитуда — 1.
    Возвращает ``(усиление, фаза в градусах)``.
    """
    x = np.asarray(signal, dtype=np.float64)
    i0, i1 = int(round(t_start * fs)), int(round(t_end * fs))
    t = np.arange(i0, i1, dtype=np.float64) / fs
    period = 1.0 / freq
    width = (i1 - i0) / fs
    if abs(width / period - round(width / period)) > 1e-9:
        raise ValueError(f"частота {freq} Гц не укладывается целым числом "
                         f"периодов в окно {width} с")
    z = x[i0:i1] * np.exp(-2j * np.pi * freq * t)
    mean = z.mean()
    gain = 2.0 * abs(mean)
    phase = float(np.degrees(np.angle(mean)))
    return float(gain), phase


__all__ = [
    "FS",
    "WAVE_LEN",
    "AGREED",
    "HAS_SECOND",
    "control_signals",
    "independent_measurement",
    "sine_gain_phase",
    "expected_gain",
    "expected_phase_rad",
]
