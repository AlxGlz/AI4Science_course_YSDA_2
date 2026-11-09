"""Проверки отбора оцениваемых волн и сравнения с эталоном (§2).

Ожидаемые значения заданы вручную: границы окна 5–55 с, положение пиков на
треугольной волне, шаг 3,125 мс и знаменатели нормировки. Тест не повторяет
код измерителя, а проверяет результат по формулировкам методики.

Данные: порождённые массивы; ``data/`` не читается.
Проверка: ``verify.py``.
Используется в: ``checks/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppg.experiment import (
    EVAL_END_SEC,
    EVAL_START_SEC,
    compare_waves,
    error_series,
    etalon_refusal_summary,
    refusal_summary,
    select_evaluation_mask,
    value_counts,
)
from ppg.io import WaveTable

FS = 320.0
START = 1600                     # 5 с
END = START + 320                # 6 с, волна целиком внутри окна
FIRST_OFFSET = 64                # фаза 0,20
SECOND_OFFSET = 192              # фаза 0,60
A1, A2 = 2.0, 1.0


def _triangle_bump(length: int, center: int, half_width: int,
                   height: float) -> np.ndarray:
    positions = np.arange(length)
    shape = np.maximum(0.0, 1.0 - np.abs(positions - center) / half_width)
    return height * shape


def _waveform(length: int = 320) -> np.ndarray:
    """Волна с двумя известными максимумами и нулевым уровнем начала."""
    return (_triangle_bump(length, FIRST_OFFSET, 50, A1)
            + _triangle_bump(length, SECOND_OFFSET, 40, A2))


def _record(wave: np.ndarray) -> np.ndarray:
    """Запись, в которой волна лежит на отрезке ``START:END+1``."""
    signal = np.zeros(START + wave.size + 1, dtype=np.float64)
    signal[START:START + wave.size] = wave
    return signal


def _waves(flags: list[bool]) -> WaveTable:
    count = len(flags)
    return WaveTable(
        start_idx=np.full(count, START, dtype=np.int64),
        end_idx=np.full(count, END, dtype=np.int64),
        has_diastolic_peak=np.asarray(flags, dtype=bool),
    )


def test_evaluation_window_bounds_are_inclusive():
    waves = WaveTable(
        start_idx=np.array([0, 1600, 1600, 1599, 1600], dtype=np.int64),
        end_idx=np.array([400, 17600, 17601, 2000, 17599], dtype=np.int64),
        has_diastolic_peak=np.ones(5, dtype=bool),
    )
    mask = select_evaluation_mask(waves, fs=FS)
    assert list(mask) == [False, True, False, False, True]
    assert EVAL_START_SEC * FS == 1600 and EVAL_END_SEC * FS == 17600


def test_compare_waves_matches_known_peaks_and_zero_error():
    signal = _record(_waveform())
    comparisons = compare_waves(_waves([True]), signal, signal)
    assert len(comparisons) == 1
    item = comparisons[0]
    # индексы пиков известны по построению
    assert item.etalon.p1 == START + FIRST_OFFSET
    assert item.etalon.p2 == START + SECOND_OFFSET
    assert item.etalon.A1 == pytest.approx(A1, abs=1e-12)
    assert item.etalon.A2 == pytest.approx(A2, abs=1e-12)
    assert item.etalon.T == pytest.approx(1000.0, abs=1e-12)
    # сравнение с самим собой даёт нулевые ошибки
    assert all(value == pytest.approx(0.0, abs=0.0)
               for value in item.absolute.values())


def test_level_shift_does_not_change_amplitudes():
    signal = _record(_waveform())
    shifted = signal + 3.0
    comparisons = compare_waves(_waves([True]), signal, shifted)
    absolute = comparisons[0].absolute
    assert absolute["A1"] == pytest.approx(0.0, abs=1e-12)
    assert absolute["A2"] == pytest.approx(0.0, abs=1e-12)
    assert absolute["t1"] == pytest.approx(0.0, abs=0.0)


def test_known_sample_shift_gives_3_125_ms_and_percent():
    """Сдвиг на один отсчёт: 3,125 мс; нормировка на T = 1000 мс → 0,3125 %."""
    signal = _record(_waveform())
    delayed = np.concatenate([np.zeros(1), signal[:-1]])
    comparisons = compare_waves(_waves([True]), signal, delayed)
    item = comparisons[0]
    assert item.estimate.p1 == START + FIRST_OFFSET + 1
    assert item.estimate.p2 == START + SECOND_OFFSET + 1
    for key in ("t1", "t2"):
        assert item.absolute[key] == pytest.approx(3.125, abs=1e-9)
        assert item.normalized[key] == pytest.approx(0.3125, abs=1e-9)
    for key in ("A1", "A2"):
        assert item.absolute[key] == pytest.approx(0.0, abs=1e-12)
        assert item.normalized[key] == pytest.approx(0.0, abs=0.0)


def test_amplitude_error_normalized_by_reference_first_peak():
    """Масштаб 1,1: ошибка A1 = 0,1·A1_э → 10 % от A1 эталона, A2 — тоже 10 %."""
    signal = _record(_waveform())
    scaled = 1.1 * signal
    item = compare_waves(_waves([True]), signal, scaled)
    comparisons = item
    assert comparisons[0].absolute["A1"] == pytest.approx(0.1 * A1, abs=1e-12)
    assert comparisons[0].normalized["A1"] == pytest.approx(10.0, abs=1e-9)
    # A2 нормируется на A1 эталона, а не на собственную амплитуду
    assert comparisons[0].absolute["A2"] == pytest.approx(0.1 * A2, abs=1e-12)
    assert comparisons[0].normalized["A2"] == pytest.approx(
        100.0 * 0.1 * A2 / A1, abs=1e-9)


def test_missing_peak_is_refusal_not_zero_error():
    signal = _record(_waveform())
    flat = np.zeros_like(signal)
    comparisons = compare_waves(_waves([True]), signal, flat)
    series = error_series(comparisons)
    normalized = error_series(comparisons, normalized=True)
    assert all(values == [None] for values in series.values())
    assert all(values == [None] for values in normalized.values())
    assert all(count == 0 for count in value_counts(series).values())

    refusals = refusal_summary(comparisons)
    assert refusals["first_peak"]["expected"] == 1
    assert refusals["first_peak"]["successes"] == 0
    assert refusals["first_peak"]["refusal_pct"] == pytest.approx(100.0)
    assert refusals["second_peak"]["refusal_pct"] == pytest.approx(100.0)


def test_second_peak_not_required_is_not_applicable():
    signal = _record(_waveform())
    flat = np.zeros_like(signal)
    comparisons = compare_waves(_waves([False]), signal, flat)
    refusals = refusal_summary(comparisons)
    assert refusals["second_peak"]["expected"] == 0
    assert refusals["second_peak"]["refusal_pct"] is None
    assert comparisons[0].absolute["A2"] is None
    assert comparisons[0].absolute["t2"] is None


def test_reference_refusals_are_counted_separately():
    signal = _record(_waveform())
    comparisons = compare_waves(_waves([True]), signal, signal)
    etalon = etalon_refusal_summary(comparisons)
    assert etalon["first_peak"]["successes"] == 1
    assert etalon["second_peak"]["successes"] == 1
    assert etalon["second_peak"]["refusal_pct"] == pytest.approx(0.0)


def test_compare_waves_rejects_bad_input():
    signal = _record(_waveform())
    with pytest.raises(ValueError, match="длины сигналов"):
        compare_waves(_waves([True]), signal, signal[:-1])
    broken = signal.copy()
    broken[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compare_waves(_waves([True]), signal, broken)
    empty = select_evaluation_mask(WaveTable(
        start_idx=np.array([], dtype=np.int64),
        end_idx=np.array([], dtype=np.int64),
        has_diastolic_peak=np.array([], dtype=bool)), fs=FS)
    with pytest.raises(ValueError, match="нет полных волн"):
        compare_waves(WaveTable(
            start_idx=np.array([], dtype=np.int64),
            end_idx=np.array([], dtype=np.int64),
            has_diastolic_peak=np.array([], dtype=bool)), signal, signal, empty)
