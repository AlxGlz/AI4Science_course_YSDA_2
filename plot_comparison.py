"""Графики сравнения сигналов по сохранённым результатам этапа (§4).

Скрипт читает ``results/<этап>/run.json``, берёт из него обработки и их
настройки, заново их не подбирает. Для каждого выбранного входа строятся
два файла в ``results/<этап>/figures/``: обзор всей записи и увеличенный
фрагмент нескольких волн. Сохраняются PNG и SVG.

Команды (из корня проекта):

    Windows: ``.\\venv\\Scripts\\python.exe plot_comparison.py --results results/test``
    macOS:   ``./.venv/bin/python plot_comparison.py --results results/test``

Выбор входа и интервала: ``--input-id``, ``--start``, ``--duration``
(время в секундах). По умолчанию — по одному входу каждого класса формы
с минимальным ID среди рассчитанных на этапе; для тестового прогона это
его единственный вход. Фильтр ``--method`` позволяет показать другие
рассчитанные варианты, если они сохранены в ``run.json`` как ``candidates``.

Данные: ``data/`` через ``ppg.io``; настройки — ``run.json`` этапа.
Проверка: ``checks/test_plotting.py``; фактические графики — в
``results/<этап>/figures/``.
Используется в: этапах §6 после расчётов, проверка результата глазами.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ppg.io import (  # noqa: E402
    CLASS_LABELS,
    SAMPLES_PER_RECORD,
    SPLIT_LABELS,
    load_dataset,
    read_signal,
)
from ppg.measure import FS_DEFAULT  # noqa: E402
from ppg.methods import apply_method  # noqa: E402
from ppg.plotting import (  # noqa: E402
    MAX_METHODS,
    build_comparison_figure,
    default_zoom,
    methods_from_report,
    save_figure,
    select_inputs,
)

STAGE_LABELS = {
    "test": "тестовый прогон",
    "select": "подбор параметров",
    "main": "основной прогон",
}


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _write_index(path: Path, stage: str,
                 run_path: Path, methods,
                 files: list[tuple[str, str, int, Path, Path, Path, Path,
                                  float, float]]) -> None:
    """Оглавление графиков этапа со ссылками на сохранённые файлы."""
    lines = [
        f"# Графики: {STAGE_LABELS[stage]}",
        "",
        "[порождено] Сгенерировано "
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}.",
        "",
        f"- Результаты этапа: `{_relative(run_path)}`.",
        f"- Обработки: "
        + ", ".join(f"«{method.label}»" for method in methods) + ".",
        "- Шкала амплитуды общая для всех панелей, без нормировки и сдвига "
        "кривых; пунктир — начало волны, круги — первый пик, "
        "квадраты — второй пик.",
        "",
        "| Вход | Класс формы | Реализация | Фрагмент, с | Обзор записи | "
        "Фрагмент |",
        "|---|---|---|---|---|---|",
    ]
    for input_id, wave_class, repeat, overview_png, overview_svg, \
            zoom_png, zoom_svg, zoom_start, zoom_duration in files:
        lines.append(
            f"| `{input_id}` | {CLASS_LABELS.get(wave_class, wave_class)} | "
            f"№{repeat} | {zoom_start:.3f}–{zoom_start + zoom_duration:.3f} | "
            f"[PNG]({overview_png.name}) · "
            f"[SVG]({overview_svg.name}) | [PNG]({zoom_png.name}) · "
            f"[SVG]({zoom_svg.name}) |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Графики сравнения обработок по результатам этапа (§4).")
    parser.add_argument("--results", type=Path, required=True,
                        help="каталог результатов этапа, например results/test")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--input-id", action="append", default=None,
                        help="вход этапа; по умолчанию по одному на класс")
    parser.add_argument("--start", type=float, default=None,
                        help="начало фрагмента, с")
    parser.add_argument("--duration", type=float, default=None,
                        help="длительность фрагмента, с")
    parser.add_argument("--method", action="append", default=None,
                        help="имя обработки; можно указать несколько раз")
    parser.add_argument("--zoom-waves", type=int, default=3,
                        help="число волн в фрагменте по умолчанию")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    results_dir = args.results.resolve()
    run_path = results_dir / "run.json"
    if not run_path.is_file():
        print(f"Остановка: нет файла результатов {run_path}. "
              f"Сначала выполните этап §6.")
        return 1
    report = json.loads(run_path.read_text(encoding="utf-8"))
    stage = str(report.get("stage") or "")
    if stage not in STAGE_LABELS:
        print(f"Остановка: неизвестный этап в run.json: {stage!r}")
        return 1

    print(f"plot_comparison.py — графики этапа «{STAGE_LABELS[stage]}» (§4)")
    print(f"Результаты: {results_dir}")
    try:
        dataset = load_dataset(args.data_dir.resolve())
        rows = select_inputs(dataset, report, args.input_id)
        methods = methods_from_report(report, args.method)
        if len(methods) > MAX_METHODS:
            raise ValueError(
                f"обработок {len(methods)}, допустимо не больше {MAX_METHODS}; "
                "укажите нужные параметром --method")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Остановка: {exc}")
        return 1

    total_sec = SAMPLES_PER_RECORD / FS_DEFAULT
    figure_dir = results_dir / "figures"
    diagnostic = (bool(report.get("diagnostics"))
                  or str(report.get("status")) in {"no_acceptable",
                                                   "diagnostic"})
    index_rows: list[tuple[str, str, int, Path, Path, Path, Path,
                           float, float]] = []
    zoom: tuple[float, float] | None = None

    for row in rows:
        reference = dataset.references[row.reference_id]
        noisy_path = dataset.data_dir / row.noisy_path
        try:
            noisy = read_signal(noisy_path, expected_samples=SAMPLES_PER_RECORD)
            processed = [(method, apply_method(noisy, method.settings, fs=FS_DEFAULT))
                         for method in methods]
            panels = []
            has_raw = any(method.settings.get("kind") == "raw"
                          for method in methods)
            if not has_raw:
                panels.append(("шумовой вход", noisy))
            for method, signal in processed:
                label = method.label
                if method.settings.get("kind") == "raw":
                    label += " (шумовой вход)"
                panels.append((label, signal))

            zoom = default_zoom(reference.waves, fs=FS_DEFAULT,
                                start=args.start, duration=args.duration,
                                waves_count=args.zoom_waves,
                                total_sec=total_sec)
            title = (f"Вход {row.input_id}: эталон {row.reference_id}, "
                     f"{CLASS_LABELS[row.wave_class]}, "
                     f"{SPLIT_LABELS[row.split]}, реализация "
                     f"№{row.noise_repeat}")
            subtitle = (f"Этап «{STAGE_LABELS[stage]}»; настройки из "
                        f"{_relative(run_path)}; общая шкала амплитуды без "
                        f"нормировки и сдвига")

            overview = build_comparison_figure(
                reference.clean, panels, reference.waves, 0.0, total_sec,
                title=title + " — обзор записи", subtitle=subtitle,
                diagnostic=diagnostic, fs=FS_DEFAULT)
            overview_png, overview_svg = save_figure(
                overview, figure_dir / f"overview_{row.input_id}",
                dpi=args.dpi)

            zoom_figure = build_comparison_figure(
                reference.clean, panels, reference.waves, zoom[0], zoom[1],
                title=title + f" — фрагмент {zoom[0]:.3f}–{zoom[0] + zoom[1]:.3f} с",
                subtitle=subtitle, diagnostic=diagnostic, fs=FS_DEFAULT)
            zoom_png, zoom_svg = save_figure(
                zoom_figure, figure_dir / f"zoom_{row.input_id}", dpi=args.dpi)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Остановка на входе {row.input_id}: {exc}")
            return 1

        index_rows.append((row.input_id, row.wave_class, row.noise_repeat,
                           overview_png, overview_svg, zoom_png, zoom_svg,
                           zoom[0], zoom[1]))
        print(f"Вход {row.input_id}: обзор и фрагмент {zoom[0]:.3f}–"
              f"{zoom[0] + zoom[1]:.3f} с — {overview_png.name}, {zoom_png.name}")

    assert zoom is not None
    _write_index(figure_dir / "index.md", stage, run_path,
                 methods, index_rows)

    elapsed = time.perf_counter() - started
    print(f"Графиков: {2 * len(rows)} файла PNG и столько же SVG; "
          f"оглавление — {_relative(figure_dir / 'index.md')}")
    print(f"Длительность: {elapsed:.2f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
