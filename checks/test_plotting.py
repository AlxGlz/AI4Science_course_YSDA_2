"""Проверки графиков ``ppg.plotting`` и правил §4.

Проверяется то, что можно установить без чтения ``data/``: выбор входов
по этапу, запрет итоговой части для графиков подбора, одна шкала амплитуды
на всех панелях, отметки начал волн и пиков, сохранение PNG и SVG.

Данные: порождённые разметки и сигналы; ``data/`` не читается.
Проверка: сам модуль; запускается ``verify.py``.
Используется в: подтверждении §4 (визуальное сравнение) перед этапами §6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg.io import Dataset, ManifestRow, WaveTable
from ppg.filters import moving_average
from ppg.plotting import (
    MAX_METHODS,
    PlotMethod,
    available_inputs,
    build_comparison_figure,
    default_inputs,
    default_zoom,
    methods_from_report,
    natural_key,
    save_figure,
    select_inputs,
)

FS = 320.0


def _row(input_id: str, reference_id: str, repeat: int, split: str,
         wave_class: str) -> ManifestRow:
    return ManifestRow(
        input_id=input_id,
        reference_id=reference_id,
        noise_repeat=repeat,
        split=split,
        wave_class=wave_class,
        clean_path=f"references/{reference_id}.csv",
        noisy_path=f"noisy/{input_id}.csv",
        annotation_path=f"annotations/{reference_id}.csv",
        dataset_version="vasotest-v2",
        clean_sha256="0" * 64,
        noisy_sha256="0" * 64,
        annotation_sha256="0" * 64,
    )


def _dataset(rows: list[ManifestRow]) -> Dataset:
    return Dataset(data_dir=Path("data"), rows=tuple(rows), references={})


def _waves(count: int = 5, length: int = 256) -> WaveTable:
    starts = np.arange(count, dtype=np.int64) * length
    ends = starts + length
    return WaveTable(start_idx=starts, end_idx=ends,
                     has_diastolic_peak=np.ones(count, dtype=bool))


def _beat(length: int = 256) -> np.ndarray:
    time = np.arange(length, dtype=np.float64)
    noise = np.random.default_rng(20261108).standard_normal(length)
    first = np.exp(-((time - 0.22 * length) ** 2) / (2 * (0.03 * length) ** 2))
    second = np.exp(-((time - 0.60 * length) ** 2) / (2 * (0.04 * length) ** 2))
    return 0.3 + 0.01 * noise + first + 0.45 * second


def test_natural_key_orders_numeric_ids():
    identifiers = ["10-1", "08-1", "02-5", "48-1"]
    assert sorted(identifiers, key=natural_key) == ["02-5", "08-1", "10-1",
                                                    "48-1"]


def test_default_inputs_take_smallest_id_per_class():
    rows = [
        _row("10-1", "10", 1, "tune", "clear"),
        _row("08-1", "08", 1, "tune", "clear"),
        _row("05-1", "05", 1, "tune", "weak"),
        _row("12-1", "12", 1, "tune", "absent"),
        _row("03-1", "03", 1, "tune", "weak"),
    ]
    chosen = default_inputs(rows)
    assert [row.input_id for row in chosen] == ["08-1", "03-1", "12-1"]
    assert [row.wave_class for row in chosen] == ["clear", "weak", "absent"]


def test_test_stage_has_single_input():
    dataset = _dataset([
        _row("08-1", "08", 1, "tune", "clear"),
        _row("08-2", "08", 2, "tune", "clear"),
    ])
    report = {"stage": "test", "test_input": {"input_id": "08-1"}}
    assert [row.input_id for row in available_inputs(dataset, report)] == [
        "08-1"]
    assert [row.input_id for row in select_inputs(dataset, report)] == ["08-1"]
    with pytest.raises(ValueError, match="не рассчитаны"):
        select_inputs(dataset, report, ["08-2"])


def test_select_stage_never_reads_holdout():
    dataset = _dataset([
        _row("08-1", "08", 1, "tune", "clear"),
        _row("20-1", "20", 1, "holdout", "clear"),
    ])
    report = {"stage": "select"}
    assert [row.input_id for row in available_inputs(dataset, report)] == [
        "08-1"]
    with pytest.raises(ValueError, match="не рассчитаны"):
        select_inputs(dataset, report, ["20-1"])
    # даже если run.json перечислил итоговый вход, подбор его не читает
    listed = {"stage": "select", "evaluated_inputs": ["20-1"]}
    with pytest.raises(ValueError, match="итоговую часть"):
        select_inputs(dataset, listed)


def test_main_stage_uses_holdout_and_honours_evaluated_inputs():
    dataset = _dataset([
        _row("08-1", "08", 1, "tune", "clear"),
        _row("20-1", "20", 1, "holdout", "clear"),
        _row("24-1", "24", 1, "holdout", "weak"),
    ])
    assert {row.input_id for row in available_inputs(dataset,
                                                     {"stage": "main"})} == {
        "20-1", "24-1"}
    report = {"stage": "main", "evaluated_inputs": ["24-1"]}
    assert [row.input_id for row in select_inputs(dataset, report)] == ["24-1"]


def test_methods_from_report_uses_selected_and_candidates():
    report = {
        "methods": [{"name": "raw", "label": "сырой сигнал",
                     "settings": {"kind": "raw"}}],
        "candidates": [
            {"name": "butterworth_0.10_20.0_N4",
             "label": "Баттерворт N=4, 0,10–20 Гц",
             "settings": {"kind": "butterworth", "f_low_hz": 0.1,
                          "f_high_hz": 20.0, "N": 4}},
        ],
    }
    default = methods_from_report(report)
    assert [method.name for method in default] == ["raw"]
    chosen = methods_from_report(report, ["butterworth_0.10_20.0_N4"])
    assert isinstance(chosen[0], PlotMethod)
    assert chosen[0].settings["N"] == 4
    with pytest.raises(ValueError, match="не рассчитаны"):
        methods_from_report(report, ["butterworth_9_9_N9"])


def test_methods_without_kind_are_rejected():
    with pytest.raises(ValueError, match="kind"):
        methods_from_report({"methods": [{"name": "broken",
                                          "settings": {"window": 11}}]})


def test_default_zoom_covers_whole_waves_in_evaluation_window():
    waves = _waves(count=75, length=256)
    start, duration = default_zoom(waves, fs=FS, waves_count=3)
    assert start == pytest.approx(1792 / FS)
    assert duration == pytest.approx((10 * 256 - 1792) / FS)
    first_start = start * FS
    assert first_start >= 5 * FS
    assert (start + duration) * FS <= 55 * FS


def test_default_zoom_accepts_explicit_interval_and_checks_bounds():
    waves = _waves(count=75, length=256)
    assert default_zoom(waves, fs=FS, start=10.0, duration=1.5) == (10.0, 1.5)
    start, duration = default_zoom(waves, fs=FS, start=10.0)
    assert start == 10.0 and duration == pytest.approx(3 * 256 / FS)
    with pytest.raises(ValueError, match="выходит за запись"):
        default_zoom(waves, fs=FS, start=59.0, duration=2.0)
    with pytest.raises(ValueError, match="положительной"):
        default_zoom(waves, fs=FS, start=10.0, duration=0.0)


def test_figure_shares_scale_and_marks_waves_and_peaks():
    waves = _waves(count=5, length=256)
    clean = np.concatenate([_beat() for _ in range(5)] + [np.array([0.3])])
    noisy = clean + 0.02 * np.sin(np.arange(clean.size) / 3.0)
    panels = [("сырой сигнал", noisy),
              ("скользящее среднее, окно 11", moving_average(noisy, 11))]
    total = clean.size / FS
    figure = build_comparison_figure(clean, panels, waves, 0.0, total,
                                     title="проверка", subtitle="синтетика",
                                     fs=FS)
    try:
        axes = figure.axes
        assert len(axes) == 1 + len(panels)
        assert len({tuple(np.round(axis.get_ylim(), 12)) for axis in axes}) == 1
        for axis in axes:
            assert axis.get_xlim() == pytest.approx((0.0, total))
            assert len(axis.lines) == 1 + len(waves)   # сигнал и начала волн
            assert len(axis.collections) >= 1          # найденные пики
        assert axes[0].get_ylabel() == "чистый эталон"
        assert "скользящее среднее" in axes[2].get_ylabel()
    finally:
        figure.clf()


def test_figure_rejects_too_many_panels_and_mismatched_lengths():
    waves = _waves(count=2, length=256)
    signal = np.concatenate([_beat() for _ in range(2)])
    panels = [(f"метод {index}", signal)
              for index in range(MAX_METHODS + 1)]
    with pytest.raises(ValueError, match="не больше"):
        build_comparison_figure(signal, panels, waves, 0.0,
                                signal.size / FS, title="t", fs=FS)
    with pytest.raises(ValueError, match="длины"):
        build_comparison_figure(signal, [("короткий", signal[:-1])], waves,
                                0.0, signal.size / FS, title="t", fs=FS)


def test_save_figure_writes_png_and_svg(tmp_path):
    waves = _waves(count=3, length=256)
    clean = np.concatenate([_beat() for _ in range(3)] + [np.array([0.3])])
    figure = build_comparison_figure(
        clean, [("сырой сигнал", clean)], waves, 0.0, clean.size / FS,
        title="проверка", fs=FS)
    png, svg = save_figure(figure, tmp_path / "figure", dpi=100)
    assert png.is_file() and svg.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"<svg" in svg.read_bytes()[:2000]
    assert png.stat().st_size > 1000
    assert svg.stat().st_size > 1000
