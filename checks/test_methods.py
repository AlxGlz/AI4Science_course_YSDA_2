"""Проверки применения обработок ``ppg.methods``.

Смысл: графики и этапы §6 должны применять ровно те же обработки, что
проверены в ``checks/test_filters.py``, и брать параметры из сохранённых
результатов, а не из своих значений. Поэтому проверяется совпадение
с ``ppg.filters``, независимость результата от входного массива и явный
отказ при неизвестном способе.

Данные: порождённые сигналы; ``data/`` не читается.
Проверка: сам модуль; запускается ``verify.py``.
Используется в: подтверждении §4 (обработка) и §6 (настройки из результатов).
"""

from __future__ import annotations

import numpy as np
import pytest

from ppg.filters import butter_bandpass, moving_average
from ppg.methods import apply_method, method_settings


def _signal(size: int = 4096) -> np.ndarray:
    time = np.arange(size) / 320.0
    return (np.sin(2.0 * np.pi * 1.2 * time)
            + 0.25 * np.sin(2.0 * np.pi * 50.0 * time)
            + 0.3 * time)


def test_raw_returns_copy_and_keeps_values():
    signal = _signal()
    result = apply_method(signal, {"kind": "raw"})
    assert result is not signal
    np.testing.assert_array_equal(result, signal)
    result[0] = 1.0e6
    assert signal[0] != 1.0e6


def test_moving_average_matches_filters():
    signal = _signal()
    for window in (5, 11, 21, 41):
        result = apply_method(signal, {"kind": "moving_average",
                                       "window": window})
        np.testing.assert_allclose(
            result, moving_average(signal, window), rtol=0.0, atol=1e-12)


def test_butterworth_matches_filters():
    signal = _signal()
    for f_low, f_high, order in ((0.3, 8.0, 2), (0.1, 20.0, 4),
                                 (0.5, 3.0, 2)):
        result = apply_method(signal, {"kind": "butterworth",
                                       "f_low_hz": f_low,
                                       "f_high_hz": f_high, "N": order})
        np.testing.assert_allclose(
            result,
            butter_bandpass(signal, f_low, f_high, order, fs=320.0),
            rtol=0.0, atol=1e-12)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="неизвестный способ"):
        apply_method(_signal(), {"kind": "median"})


def test_method_settings_keeps_only_parameters():
    method = {"name": "butterworth_0.30_8.0_N2",
              "label": "Баттерворт N=2, 0,30–8 Гц",
              "settings": {"kind": "butterworth", "f_low_hz": 0.3,
                           "f_high_hz": 8.0, "N": 2}}
    assert method_settings(method["settings"]) == method["settings"]
    assert method_settings({"kind": "moving_average", "window": 11,
                            "label": "среднее"}) == {"kind": "moving_average",
                                                     "window": 11}
    with pytest.raises(ValueError, match="kind"):
        method_settings({"window": 11})


@pytest.mark.parametrize("bad", [
    np.array([0.0, np.nan, 1.0]),
    np.array([0.0, np.inf, 1.0]),
    np.zeros((4, 3)),
    np.array([]),
])
def test_invalid_signal_is_rejected(bad):
    with pytest.raises(ValueError):
        apply_method(bad, {"kind": "raw"})


def test_even_window_is_rejected():
    with pytest.raises(ValueError, match="нечётным"):
        apply_method(_signal(), {"kind": "moving_average", "window": 10})
