"""Проверки измерения пиков по §2 и контрольным сигналам §5.

Ожидаемые ответы берутся из независимого перебора ``design.controls`` и из
согласованной таблицы ``checks/reference.py::AGREED``, а не из проверяемого
``src/ppg``.

Данные: контрольные сигналы порождаются формулами, файлы не читаются.
Проверка: ``verify.py``.
Используется в: ``verify.py`` (обязательные проверки).
"""

from __future__ import annotations

import numpy as np
import pytest

from ppg import measure_wave
from ppg.measure import FS_DEFAULT
from reference import (
    AGREED,
    HAS_SECOND,
    WAVE_LEN,
    control_signals,
    independent_measurement,
)
from design import controls

ABS_TOL = 1e-9          # допуск §5 на амплитуды и времена


def _assert_measurement(m: object, expected: dict) -> None:
    """Сверить измерение с ожидаемым словарём (индексы точно, значения 1e-9)."""
    for key in ("p1", "p2"):
        if expected[key] is None:
            assert m.p1 is None if key == "p1" else m.p2 is None
        else:
            got = m.p1 if key == "p1" else m.p2
            assert got == expected[key], (key, got, expected[key])
    for key in ("A1", "A2", "t1", "t2"):
        if expected[key] is None:
            assert getattr(m, key) is None, (key, getattr(m, key))
        else:
            got = getattr(m, key)
            assert got is not None
            assert abs(got - expected[key]) <= ABS_TOL, (key, got, expected[key])


def test_control_signals_match_method() -> None:
    """CTRL-W1…W6b: независимый перебор совпадает с таблицей §5 и с измерителем."""
    signals = control_signals()
    assert set(signals) == set(AGREED)
    for name, x in signals.items():
        expected = AGREED[name]
        independent = independent_measurement(x, HAS_SECOND[name])
        # 1) независимый перебор против согласованной таблицы
        for key in ("p1", "p2", "A1", "A2", "t1", "t2"):
            if expected[key] is None:
                assert independent[key] is None, (name, key, independent[key])
            else:
                assert independent[key] is not None, (name, key)
                assert abs(independent[key] - expected[key]) <= ABS_TOL, (
                    name, key, independent[key], expected[key])
        # 2) измеритель проекта против согласованной таблицы
        m = measure_wave(x, 0, WAVE_LEN, HAS_SECOND[name])
        _assert_measurement(m, expected)
        assert m.T == pytest.approx(1000.0, abs=ABS_TOL)
        assert m.second_required is HAS_SECOND[name]
        # отказы: W2 и W4 размечены, но пика в окне нет; W3 — неприменим
        assert m.refused_second is (name in {"CTRL-W2", "CTRL-W4"}), name
        assert m.refused_first is False, name


def test_known_two_peaks() -> None:
    """Два известных пика: A1=2, A2=1, t1=200, t2=600; плечо — отказ."""
    x = controls.two_peak_wave()
    m = measure_wave(x, 0, WAVE_LEN, True)
    assert (m.p1, m.p2) == (64, 192)
    assert m.A1 == pytest.approx(2.0, abs=ABS_TOL)
    assert m.A2 == pytest.approx(1.0, abs=ABS_TOL)
    assert m.t1 == pytest.approx(200.0, abs=ABS_TOL)
    assert m.t2 == pytest.approx(600.0, abs=ABS_TOL)

    shoulder = measure_wave(controls.shoulder_wave(), 0, WAVE_LEN, True)
    assert shoulder.p1 == 64
    assert shoulder.A1 == pytest.approx(2.0, abs=ABS_TOL)
    assert shoulder.p2 is None and shoulder.refused_second is True

    single = measure_wave(controls.single_peak_wave(), 0, WAVE_LEN, False)
    assert single.p2 is None
    assert single.second_required is False
    assert single.refused_second is False


def test_flat_top_middle_rounded_down() -> None:
    """Плоская вершина 62..65 даёт отсчёт 63 (округление вниз), не 64."""
    m = measure_wave(controls.flat_top_wave(), 0, WAVE_LEN, True)
    assert m.p1 == 63
    assert m.A1 == pytest.approx(1.99, abs=ABS_TOL)
    assert m.t1 == pytest.approx(196.875, abs=ABS_TOL)


def test_second_peak_chosen_by_prominence() -> None:
    """Второй пик выбирается по prominence (240), а не по высоте (205)."""
    m = measure_wave(controls.prominence_wave(), 0, WAVE_LEN, True)
    assert m.p2 == 240
    assert m.A2 == pytest.approx(0.8, abs=ABS_TOL)
    assert m.t2 == pytest.approx(750.0, abs=ABS_TOL)


@pytest.mark.parametrize("offset", [-1000.0, -0.7, 0.0, 0.7, 1000.0])
def test_level_shift_invariance(offset: float) -> None:
    """Добавление постоянной не меняет индексы, амплитуды и времена (§5)."""
    x = controls.two_peak_wave()
    base = measure_wave(x, 0, WAVE_LEN, True)
    shifted = measure_wave(x + offset, 0, WAVE_LEN, True)
    assert (base.p1, base.p2) == (shifted.p1, shifted.p2)
    for key in ("A1", "A2", "t1", "t2", "T"):
        assert getattr(shifted, key) == pytest.approx(getattr(base, key),
                                                      abs=ABS_TOL), key


def test_scale_invariance_and_error_scaling() -> None:
    """Умножение на 3: амплитуды и |ΔA| ×3, времена и нормированные — нет."""
    from ppg.metrics import absolute_errors, normalized_errors

    # эталон и «оценка» с другими высотами пиков
    n = WAVE_LEN + 1
    ref = np.full(n, 0.3)
    ref += controls.cos_bump(n, 64.0, 32.0, 2.0)
    ref += controls.cos_bump(n, 192.0, 32.0, 1.0)
    est = np.full(n, 0.3)
    est += controls.cos_bump(n, 64.0, 32.0, 2.2)
    est += controls.cos_bump(n, 192.0, 32.0, 0.9)

    for k in (1.0, 3.0, 0.25):
        ref_k = measure_wave(k * ref, 0, WAVE_LEN, True)
        est_k = measure_wave(k * est, 0, WAVE_LEN, True)
        base_ref = measure_wave(ref, 0, WAVE_LEN, True)
        base_est = measure_wave(est, 0, WAVE_LEN, True)

        for key in ("p1", "p2"):
            assert getattr(ref_k, key) == getattr(base_ref, key)
            assert getattr(est_k, key) == getattr(base_est, key)
        for key in ("A1", "A2"):
            assert getattr(ref_k, key) == pytest.approx(
                k * getattr(base_ref, key), rel=1e-12)
        for key in ("t1", "t2", "T"):
            assert getattr(ref_k, key) == pytest.approx(
                getattr(base_ref, key), abs=ABS_TOL)

        abs_k = absolute_errors(ref_k, est_k)
        abs_1 = absolute_errors(base_ref, base_est)
        for key in ("A1", "A2"):
            assert abs_k[key] == pytest.approx(k * abs_1[key], rel=1e-12), key
        for key in ("t1", "t2"):
            assert abs_k[key] == pytest.approx(abs_1[key], abs=ABS_TOL), key
        norm_k = normalized_errors(ref_k, est_k)
        norm_1 = normalized_errors(base_ref, base_est)
        for key in ("A1", "t1", "A2", "t2"):
            assert norm_k[key] == pytest.approx(norm_1[key], abs=ABS_TOL), key


@pytest.mark.parametrize("shift,expected_t1", [(4, 212.5), (-4, 187.5)])
def test_known_time_shift(shift: int, expected_t1: float) -> None:
    """Смещение ±4 отсчёта: Δt = 12,5 мс, 1,25 %, |ΔA| = 0 (§5)."""
    from ppg.metrics import absolute_errors, normalized_errors

    base = measure_wave(controls.two_peak_wave(), 0, WAVE_LEN, True)
    moved = measure_wave(controls.shifted_first_peak_wave(shift), 0, WAVE_LEN, True)

    assert moved.p1 == 64 + shift
    assert moved.t1 == pytest.approx(expected_t1, abs=ABS_TOL)
    assert moved.A1 == pytest.approx(2.0, abs=ABS_TOL)

    abs_err = absolute_errors(base, moved)
    norm_err = normalized_errors(base, moved)
    assert abs_err["t1"] == pytest.approx(12.5, abs=ABS_TOL)
    assert norm_err["t1"] == pytest.approx(1.25, abs=ABS_TOL)
    assert abs_err["A1"] == pytest.approx(0.0, abs=ABS_TOL)
    assert abs_err["A2"] == pytest.approx(0.0, abs=ABS_TOL)
    assert abs_err["t2"] == pytest.approx(0.0, abs=ABS_TOL)


def test_empty_windows_are_refusals() -> None:
    """Нет локального максимума в окне — отказ, а не нулевая ошибка."""
    n = WAVE_LEN + 1
    x = np.full(n, 0.3)                      # ровный сигнал: максимумов нет
    x += controls.cos_bump(n, 64.0, 32.0, 2.0)
    # в окне первого пика максимум есть; убираем его из окна 0.40..0.85
    m = measure_wave(x, 0, WAVE_LEN, True)
    assert m.p1 == 64
    assert m.p2 is None and m.refused_second is True
    assert m.A2 is None and m.t2 is None


def test_samples_to_ms_uses_320_hz() -> None:
    """Шаг сетки 3,125 мс: t(1 отсчёт) = 3,125 мс, T(320) = 1000 мс."""
    x = controls.two_peak_wave()
    m = measure_wave(x, 0, WAVE_LEN, True)
    assert FS_DEFAULT == 320.0
    assert m.T == pytest.approx(320 * 3.125, abs=ABS_TOL)
    assert 1000.0 * (m.p1 - 0) / 320 == pytest.approx(m.t1, abs=ABS_TOL)
