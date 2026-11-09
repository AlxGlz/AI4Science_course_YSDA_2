"""Графики сравнения обработок по §4 методики.

Визуальное сравнение показывает на общей временной шкале чистый эталон и
обработки, параметры которых взяты из ``run.json`` этапа, а не подобраны
заново (§4). Оси времени общие, шкала амплитуды на всех панелях одна и та же:
кривые не нормируются и не сдвигаются. На панелях отмечаются начала волн
(пунктир) и найденные измерителем пики (круги — первый, квадраты — второй).

Правила выбора входов (§4):

* этап ``test`` — единственный вход из ``run.json``;
* этап ``select`` — только данные для подбора (``tune``), итоговая часть
  не читается;
* этап ``main`` — данные для итоговой проверки (``holdout``);
* по умолчанию — по одному входу каждого класса формы с минимальным ID;
* ``evaluated_inputs`` в ``run.json`` ограничивает набор фактически
  рассчитанными входами.

Данные: ``data/`` через ``ppg.io``; настройки — ``run.json`` этапа.
Проверка: ``checks/test_plotting.py`` (выбор входов и запрет итоговой части
для подбора, общие шкалы, отметки пиков, сохранение PNG и SVG).
Используется в: ``plot_comparison.py``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .io import WAVE_CLASSES, Dataset, ManifestRow, WaveTable  # noqa: E402
from .measure import FS_DEFAULT, measure_wave  # noqa: E402
from .methods import method_settings  # noqa: E402

STAGE_SPLITS = {"test": "tune", "select": "tune", "main": "holdout"}
MAX_METHODS = 6


@dataclass(frozen=True)
class PlotMethod:
    """Обработка для графика: имя, подпись и параметры из результатов."""

    name: str
    label: str
    settings: dict[str, object]


def natural_key(text: str) -> list[object]:
    """Ключ сортировки, при котором ``08`` идёт раньше ``10``."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def _plot_method(item: Mapping[str, object]) -> PlotMethod:
    name = str(item.get("name") or "")
    if not name:
        raise ValueError(f"у метода нет имени: {item!r}")
    label = str(item.get("label") or name)
    raw_settings = item.get("settings")
    settings = (dict(raw_settings) if isinstance(raw_settings, Mapping)
                else method_settings(item))
    method_settings(settings)          # проверка обязательного поля kind
    return PlotMethod(name=name, label=label, settings=settings)


def methods_from_report(report: Mapping[str, object],
                        requested: Sequence[str] | None = None
                        ) -> list[PlotMethod]:
    """Выбрать обработки из ``run.json``; ``requested`` — по именам (§4).

    По умолчанию берутся выбранные методы этапа (``methods``). Если в
    результатах подбора сохранены все рассчитанные варианты (``candidates``),
    параметром ``--method`` можно показать любой из них.
    """
    methods = [_plot_method(item) for item in report.get("methods", ())]
    candidates = [_plot_method(item) for item in report.get("candidates", ())]
    available: dict[str, PlotMethod] = {item.name: item for item in candidates}
    available.update({item.name: item for item in methods})
    if not available:
        raise ValueError("в результатах этапа нет описаний обработок")
    if requested:
        names = list(dict.fromkeys(str(name) for name in requested))
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(
                f"обработки не рассчитаны на этом этапе: {missing}; "
                f"доступны: {sorted(available)}")
        return [available[name] for name in names]
    if not methods:
        raise ValueError("в результатах этапа нет выбранных обработок")
    return methods


def available_inputs(dataset: Dataset,
                     report: Mapping[str, object]) -> list[ManifestRow]:
    """Входы, рассчитанные на этапе, по ``run.json`` и части выборки (§4)."""
    listed = report.get("evaluated_inputs")
    if listed:
        identifiers = [
            str(item["input_id"]) if isinstance(item, Mapping) else str(item)
            for item in listed]
        return [dataset.by_input(identifier) for identifier in identifiers]

    stage = str(report.get("stage") or "")
    if stage == "test":
        test_input = report.get("test_input")
        if not isinstance(test_input, Mapping) or "input_id" not in test_input:
            raise ValueError("в результатах тестового прогона нет test_input")
        return [dataset.by_input(str(test_input["input_id"]))]

    split = STAGE_SPLITS.get(stage)
    if split is None:
        raise ValueError(f"неизвестный этап результатов: {stage!r}")
    rows = [row for row in dataset.rows if row.split == split]
    if not rows:
        raise ValueError(f"нет входов части {split!r} для этапа {stage!r}")
    return rows


def default_inputs(rows: Sequence[ManifestRow]) -> list[ManifestRow]:
    """По одному входу каждого класса формы с минимальным ID (§4)."""
    chosen: dict[str, ManifestRow] = {}
    for row in sorted(rows, key=lambda item: natural_key(item.input_id)):
        chosen.setdefault(row.wave_class, row)
    ordered = [chosen[wave_class] for wave_class in WAVE_CLASSES
               if wave_class in chosen]
    if not ordered:
        raise ValueError("нет входов для построения графиков")
    return ordered


def select_inputs(dataset: Dataset, report: Mapping[str, object],
                  input_ids: Sequence[str] | None = None) -> list[ManifestRow]:
    """Выбрать входы для графиков по §4.

    Явно указанные ``input_ids`` должны быть рассчитаны на этапе. Для
    подбора запрещена итоговая часть: её нельзя читать до фиксации настроек.
    """
    available = available_inputs(dataset, report)
    by_id = {row.input_id: row for row in available}
    if input_ids:
        identifiers = list(dict.fromkeys(str(value) for value in input_ids))
        missing = [value for value in identifiers if value not in by_id]
        if missing:
            raise ValueError(
                f"входы не рассчитаны на этапе: {missing}; "
                f"доступны: {sorted(by_id, key=natural_key)}")
        rows = [by_id[value] for value in identifiers]
    else:
        rows = default_inputs(available)

    if report.get("stage") == "select":
        holdout = [row.input_id for row in rows if row.split != "tune"]
        if holdout:
            raise ValueError(
                "графики подбора не должны читать итоговую часть: "
                + ", ".join(holdout))
    return rows


def default_zoom(waves: WaveTable, fs: float = FS_DEFAULT,
                 start: float | None = None, duration: float | None = None,
                 waves_count: int = 3,
                 total_sec: float = 60.0) -> tuple[float, float]:
    """Интервал увеличенного фрагмента: ``waves_count`` волн или явные границы.

    Без аргументов фрагмент начинается с первой полной волны окна 5–55 с
    и содержит ``waves_count`` волн. Проверяется, что интервал положителен
    и не выходит за запись.
    """
    if int(waves_count) != waves_count or waves_count < 1:
        raise ValueError(f"число волн фрагмента должно быть >= 1: {waves_count}")
    starts = np.asarray(waves.start_idx, dtype=np.int64)
    ends = np.asarray(waves.end_idx, dtype=np.int64)
    if start is None and duration is None:
        mask = (starts >= 5.0 * fs) & (ends <= 55.0 * fs)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise ValueError("в окне 5–55 с нет полных волн для фрагмента")
        chosen = indices[:min(int(waves_count), indices.size)]
        start = float(starts[chosen[0]] / fs)
        duration = float((ends[chosen[-1]] - starts[chosen[0]]) / fs)
    elif start is None:
        start = 5.0
    elif duration is None:
        lengths = (ends - starts) / fs
        duration = float(waves_count * float(np.median(lengths)))

    start = float(start)
    duration = float(duration)
    if not np.isfinite(start) or not np.isfinite(duration):
        raise ValueError("интервал фрагмента должен быть конечным")
    if start < 0:
        raise ValueError(f"начало фрагмента отрицательно: {start}")
    if duration <= 0:
        raise ValueError(f"длительность фрагмента должна быть положительной: {duration}")
    if start + duration > total_sec + 1e-9:
        raise ValueError(
            f"интервал {start:g}–{start + duration:g} с выходит за запись "
            f"{total_sec:g} с")
    return start, duration


def peaks_in_interval(signal: np.ndarray, waves: WaveTable, start_sec: float,
                      duration_sec: float,
                      fs: float = FS_DEFAULT) -> list[dict[str, object]]:
    """Начала волн и найденные пики для панели графика (§2).

    Пики измеряются тем же измерителем, что и в эксперименте, на полной
    волне; возвращаются только волны, пересекающие интервал отображения.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("сигнал панели должен быть одномерным")
    end_sec = start_sec + duration_sec
    peaks: list[dict[str, object]] = []
    for index in range(len(waves)):
        s, b, flag = waves.wave(index)
        if b / fs < start_sec or s / fs > end_sec:
            continue
        measurement = measure_wave(x, s, b, flag, fs=fs)
        peaks.append({
            "wave_index": index,
            "start_idx": s,
            "p1": measurement.p1,
            "p2": measurement.p2,
        })
    return peaks


def build_comparison_figure(etalon: np.ndarray,
                            panels: Sequence[tuple[str, np.ndarray]],
                            waves: WaveTable, start_sec: float,
                            duration_sec: float, *,
                            title: str, subtitle: str = "",
                            diagnostic: bool = False,
                            fs: float = FS_DEFAULT) -> Figure:
    """Собрать фигуру: эталон и обработки на общей шкале (§4).

    ``panels`` — пары ``(подпись с настройками, сигнал)``. Шкала амплитуды
    одна для всех панелей и строится по данным интервала без нормировки
    и сдвига; ось времени общая.
    """
    if not panels:
        raise ValueError("нет панелей обработок для графика")
    if len(panels) > MAX_METHODS:
        raise ValueError(
            f"панелей {len(panels)}, допустимо не больше {MAX_METHODS}; "
            "укажите нужные обработки параметром --method")
    etalon = np.asarray(etalon, dtype=np.float64)
    if etalon.ndim != 1 or etalon.size == 0:
        raise ValueError("эталонный сигнал должен быть непустым и одномерным")
    signals = [etalon]
    labels = ["чистый эталон"]
    for label, signal in panels:
        values = np.asarray(signal, dtype=np.float64)
        if values.shape != etalon.shape:
            raise ValueError(
                f"длины сигналов различаются: {values.size} и {etalon.size}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"панель {label!r} содержит NaN или бесконечность")
        signals.append(values)
        labels.append(str(label))

    fs = float(fs)
    low = max(0, int(np.floor(start_sec * fs)))
    high = min(etalon.size, int(np.ceil((start_sec + duration_sec) * fs)) + 1)
    if high - low < 2:
        raise ValueError("в интервале графика меньше двух отсчётов")
    window_values = np.concatenate([signal[low:high] for signal in signals])
    span = float(np.max(window_values) - np.min(window_values))
    margin = 0.05 * span if span > 0 else 1.0
    y_limits = (float(np.min(window_values) - margin),
                float(np.max(window_values) + margin))

    figure, axes = plt.subplots(
        len(signals), 1, sharex=True, sharey=True,
        figsize=(11.0, 1.5 * len(signals) + 1.1), squeeze=False)
    axes = axes[:, 0]
    time = np.arange(etalon.size) / fs

    for axis, label, signal in zip(axes, labels, signals):
        peaks = peaks_in_interval(signal, waves, start_sec, duration_sec, fs=fs)
        axis.plot(time, signal, color="0.25", lw=0.9)
        for number, peak in enumerate(peaks):
            axis.axvline(
                float(peak["start_idx"]) / fs, color="0.55", lw=0.6,
                ls=":", zorder=1,
                label="начало волны" if number == 0 else None)
        first = [(int(peak["p1"]), signal[int(peak["p1"])])
                 for peak in peaks if peak["p1"] is not None]
        second = [(int(peak["p2"]), signal[int(peak["p2"])])
                  for peak in peaks if peak["p2"] is not None]
        if first:
            axis.scatter([value[0] / fs for value in first],
                         [value[1] for value in first], s=14,
                         color="#c0392b", zorder=3, label="первый пик")
        if second:
            axis.scatter([value[0] / fs for value in second],
                         [value[1] for value in second], s=12,
                         marker="s", color="#1f6fb4", zorder=3,
                         label="второй пик")
        axis.set_ylabel(label, fontsize=8)
        axis.set_xlim(start_sec, start_sec + duration_sec)
        axis.set_ylim(*y_limits)
        axis.grid(True, color="0.92", lw=0.6)
        axis.tick_params(labelsize=8)

    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    axes[-1].set_xlabel("время, с", fontsize=9)
    heading = title + (" — диагностические" if diagnostic else "")
    figure.suptitle(heading, fontsize=10)
    if subtitle:
        figure.text(0.01, 0.005, subtitle, fontsize=7, color="0.35",
                    ha="left", va="bottom")
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    return figure


def save_figure(figure: Figure, stem: str | Path,
                dpi: int = 150) -> tuple[Path, Path]:
    """Сохранить фигуру в PNG и SVG рядом с ``stem`` и закрыть её (§4)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    figure.savefig(png, dpi=int(dpi))
    figure.savefig(svg)
    plt.close(figure)
    return png, svg
