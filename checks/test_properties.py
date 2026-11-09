"""Проверки свойств на порождённых входах (§4, §5).

Профиль Hypothesis зафиксирован с ``derandomize=True`` (§4): при одном и том
же коде и версиях пакетов примеры воспроизводятся. Свойства — смещение
уровня, масштабирование амплитуд, перенос пика во времени, независимость
усреднения от порядка строк.

Данные: сигналы порождаются формулами; файлы не читаются.
Проверка: ``verify.py``.
Используется в: ``verify.py`` (обязательная проверка уровня).
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ppg import measure_wave
from ppg.metrics import nested_mean
from design import controls

FS = 320.0
WAVE_LEN = 320
ABS_TOL = 1e-9

settings.register_profile("deterministic", derandomize=True, max_examples=60,
                          deadline=None)
settings.load_profile("deterministic")


@given(offset=st.floats(min_value=-1e3, max_value=1e3,
                        allow_nan=False, allow_infinity=False))
def test_level_shift_property(offset: float) -> None:
    """Добавление постоянной не меняет пики и амплитуды относительно начала."""
    base = measure_wave(controls.two_peak_wave(), 0, WAVE_LEN, True)
    shifted = measure_wave(controls.two_peak_wave() + offset, 0, WAVE_LEN, True)
    assert (base.p1, base.p2) == (shifted.p1, shifted.p2)
    for key in ("A1", "A2", "t1", "t2", "T"):
        assert getattr(shifted, key) == pytest.approx(getattr(base, key),
                                                      abs=ABS_TOL), key


@given(k=st.floats(min_value=1e-3, max_value=1e4, allow_nan=False,
                   allow_infinity=False))
def test_amplitude_scale_property(k: float) -> None:
    """Умножение на k > 0 масштабирует амплитуды, не трогая времена."""
    base = measure_wave(controls.two_peak_wave(), 0, WAVE_LEN, True)
    scaled = measure_wave(k * controls.two_peak_wave(), 0, WAVE_LEN, True)
    assert (base.p1, base.p2) == (scaled.p1, scaled.p2)
    for key in ("A1", "A2"):
        assert getattr(scaled, key) == pytest.approx(
            k * getattr(base, key), rel=1e-12), key
    for key in ("t1", "t2", "T"):
        assert getattr(scaled, key) == pytest.approx(getattr(base, key),
                                                     abs=ABS_TOL), key


@given(shift=st.integers(min_value=-20, max_value=20))
def test_time_shift_property(shift: int) -> None:
    """Сдвиг первого бугра на целое число отсчётов смещает t1 ровно на столько."""
    base = measure_wave(controls.two_peak_wave(), 0, WAVE_LEN, True)
    moved = measure_wave(controls.shifted_first_peak_wave(shift), 0, WAVE_LEN, True)
    assert moved.p1 == base.p1 + shift
    assert moved.A1 == pytest.approx(base.A1, abs=ABS_TOL)
    assert moved.A2 == pytest.approx(base.A2, abs=ABS_TOL)
    assert moved.p2 == base.p2
    expected_t1 = 1000.0 * (moved.p1 - 0) / FS
    assert moved.t1 == pytest.approx(expected_t1, abs=ABS_TOL)


@given(values=st.lists(st.floats(min_value=0.0, max_value=50.0,
                                 allow_nan=False, allow_infinity=False),
                       min_size=1, max_size=6),
       seed=st.integers(min_value=0, max_value=10**6))
def test_averaging_order_property(values: list[float], seed: int) -> None:
    """Порядок волн внутри реализации не влияет на среднее с равными весами."""
    groups = {"E1": {"r0": list(values)}, "E2": {"r0": [3.0]}}
    reference = nested_mean(groups)
    rng = random.Random(seed)
    shuffled = list(values)
    rng.shuffle(shuffled)
    assert nested_mean({"E1": {"r0": shuffled}, "E2": {"r0": [3.0]}}) == \
        pytest.approx(reference, abs=ABS_TOL)
