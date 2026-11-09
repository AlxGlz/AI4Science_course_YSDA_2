"""Этапы эксперимента §6 методики.

Реализован этап ``prepare``: проверка данных и верификатора и сохранение
плана расчётов в ``results/prepare/``. Экспериментальные вычисления на этом
этапе не выполняются (§6).

Команды (из корня проекта):

    Windows: ``.\\venv\\Scripts\\python.exe run_experiment.py prepare``
    macOS:   ``./.venv/bin/python run_experiment.py prepare``

Данные: ``data/manifest.csv``, ``data/references/``, ``data/noisy/``,
``data/annotations/``; чтение — через ``ppg.io``, без вычисления пиков.
Проверка: ``verify.py`` (запускается этим скриптом первым шагом) и
``results/prepare/``.
Используется в: этапы ``test``, ``select``, ``main`` после реализации.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ppg.io import (  # noqa: E402
    CLASS_LABELS,
    SAMPLES_PER_RECORD,
    SPLIT_LABELS,
    SPLITS,
    load_dataset,
    read_signal,
    sha256_file,
)

FILTER_LOW_HZ = (0.10, 0.30, 0.50)
FILTER_HIGH_HZ = (3.0, 5.0, 8.0, 12.0, 20.0)
FILTER_ORDERS = (2, 4)
MOVING_WINDOWS = (5, 11, 21, 41)
DATASET_VERSION = "vasotest-v2"
EXPECTED_REFERENCES = 48
EXPECTED_INPUTS = 240
EXPECTED_PER_SPLIT = 24
EXPECTED_PER_CLASS = 8


def natural_key(text: str) -> list[object]:
    """Ключ сортировки, при котором ``08`` идёт раньше ``10``."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def filter_grid() -> list[dict[str, object]]:
    """30 сочетаний §2: 3 нижние × 5 верхних частот × 2 порядка."""
    return [
        {"f_low_hz": f_low, "f_high_hz": f_high, "N": order}
        for f_low in FILTER_LOW_HZ
        for f_high in FILTER_HIGH_HZ
        for order in FILTER_ORDERS
    ]


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "matplotlib", "statsmodels", "pytest",
                    "hypothesis"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "не установлен"
    return versions


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "неизвестен"


def _python_command() -> str:
    """Команда интерпретатора из корня проекта для записи в план."""
    executable = Path(sys.executable).resolve()
    try:
        return str(executable.relative_to(ROOT))
    except ValueError:
        return str(executable)


def _write_csv(path: Path, fieldnames: list[str],
               rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _check_references(dataset, problems: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for reference_id in sorted(dataset.references, key=natural_key):
        reference = dataset.references[reference_id]
        local: list[str] = []
        clean_exists = reference.clean_path.is_file()
        clean_sha_ok = False
        if clean_exists:
            clean_sha_ok = (
                sha256_file(reference.clean_path)
                == reference.inputs[0].clean_sha256)
            if not clean_sha_ok:
                local.append("SHA-256 чистого эталона не совпал")
        else:
            local.append("нет файла чистого эталона")

        annotation_exists = reference.annotation_path.is_file()
        annotation_sha_ok = False
        if annotation_exists:
            annotation_sha_ok = (
                sha256_file(reference.annotation_path)
                == reference.inputs[0].annotation_sha256)
            if not annotation_sha_ok:
                local.append("SHA-256 разметки не совпал")
        else:
            local.append("нет файла разметки")

        if reference.clean.shape != (SAMPLES_PER_RECORD,):
            local.append(f"чистый сигнал {reference.clean.shape[0]} отсчётов")
        if len(reference.waves) == 0:
            local.append("разметка пуста")

        repeats = sorted(row.noise_repeat for row in reference.inputs)
        if repeats != [1, 2, 3, 4, 5]:
            local.append(f"повторы {repeats} вместо 1..5")

        flagged = int(reference.waves.has_diastolic_peak.sum())
        if reference.wave_class == "absent" and flagged:
            local.append(f"у класса absent {flagged} волн с вторым пиком")

        if local:
            problems.extend(f"{reference_id}: {text}" for text in local)
        rows.append({
            "reference_id": reference_id,
            "split": reference.split,
            "split_label": SPLIT_LABELS[reference.split],
            "wave_class": reference.wave_class,
            "wave_class_label": CLASS_LABELS[reference.wave_class],
            "clean_exists": clean_exists,
            "clean_sha_ok": clean_sha_ok,
            "samples": int(reference.clean.shape[0]),
            "annotation_exists": annotation_exists,
            "annotation_sha_ok": annotation_sha_ok,
            "waves": len(reference.waves),
            "flagged_waves": flagged,
            "last_end_idx": int(reference.waves.end_idx[-1]),
            "ok": not local,
            "problems": "; ".join(local),
        })
    return rows


def _check_inputs(dataset, problems: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data_dir = dataset.data_dir
    for row in sorted(dataset.rows, key=lambda item: natural_key(item.input_id)):
        local: list[str] = []
        path = (data_dir / row.noisy_path).resolve()
        exists = path.is_file()
        sha_ok = False
        samples: int | str = "—"
        finite = False
        if exists:
            sha_ok = sha256_file(path) == row.noisy_sha256
            if not sha_ok:
                local.append("SHA-256 шумового входа не совпал")
            try:
                signal = read_signal(path, expected_samples=SAMPLES_PER_RECORD)
                samples = int(signal.size)
                finite = True
            except ValueError as exc:
                local.append(str(exc))
        else:
            local.append("нет файла шумового входа")

        if local:
            problems.extend(f"{row.input_id}: {text}" for text in local)
        rows.append({
            "input_id": row.input_id,
            "reference_id": row.reference_id,
            "noise_repeat": row.noise_repeat,
            "split": row.split,
            "split_label": SPLIT_LABELS[row.split],
            "wave_class": row.wave_class,
            "wave_class_label": CLASS_LABELS[row.wave_class],
            "noisy_path": row.noisy_path,
            "exists": exists,
            "sha256_ok": sha_ok,
            "samples": samples,
            "finite": finite,
            "ok": not local,
            "problems": "; ".join(local),
        })
    return rows


def _contract_problems(dataset) -> list[str]:
    problems: list[str] = []
    if len(dataset.rows) != EXPECTED_INPUTS:
        problems.append(
            f"входов {len(dataset.rows)}, ожидалось {EXPECTED_INPUTS}")
    if len(dataset.references) != EXPECTED_REFERENCES:
        problems.append(
            f"эталонов {len(dataset.references)}, ожидалось {EXPECTED_REFERENCES}")

    for split in SPLITS:
        references = {row.reference_id for row in dataset.rows
                      if row.split == split}
        if len(references) != EXPECTED_PER_SPLIT:
            problems.append(
                f"{split}: эталонов {len(references)}, "
                f"ожидалось {EXPECTED_PER_SPLIT}")
        counts = Counter(dataset.references[ref].wave_class for ref in references)
        for wave_class in CLASS_LABELS:
            if counts[wave_class] != EXPECTED_PER_CLASS:
                problems.append(
                    f"{split}/{wave_class}: эталонов {counts[wave_class]}, "
                    f"ожидалось {EXPECTED_PER_CLASS}")

    versions = {row.dataset_version for row in dataset.rows}
    if versions != {DATASET_VERSION}:
        problems.append(f"версии набора {sorted(versions)}")
    return problems


def _choose_test_input(dataset):
    """Первый по ID эталон класса ``clear``, реализация №1 (§6)."""
    candidates = [
        row for row in dataset.rows
        if row.split == "tune" and row.wave_class == "clear"
        and row.noise_repeat == 1
    ]
    if not candidates:
        raise ValueError("нет входа класса clear в части tune для теста")
    return min(candidates, key=lambda row: natural_key(row.reference_id))


def _run_verifier(out_dir: Path, say) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "verify.py")],
        cwd=str(ROOT), capture_output=True, text=True)
    text = result.stdout
    if result.stderr:
        text += "\n[stderr]\n" + result.stderr
    (out_dir / "verify.log").write_text(text, encoding="utf-8")
    passed = next((line.strip() for line in reversed(text.splitlines())
                   if "passed" in line or "failed" in line), "—")
    say(f"Верификатор: код {result.returncode}; {passed}")
    return result.returncode, passed


def _split_summary(dataset) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        references = {row.reference_id for row in dataset.rows
                      if row.split == split}
        classes = Counter(dataset.references[ref].wave_class for ref in references)
        summary[split] = {
            "label": SPLIT_LABELS[split],
            "references": len(references),
            "inputs": sum(1 for row in dataset.rows if row.split == split),
            "classes": dict(sorted(classes.items())),
        }
    return summary


def run_prepare(data_dir: Path, results_dir: Path) -> int:
    """Этап ``prepare``: верификатор, проверка данных, план расчётов."""
    started = time.perf_counter()
    out_dir = results_dir / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def say(message: str = "") -> None:
        print(message)
        log.append(message)

    say("run_experiment.py prepare — план расчётов и проверка данных (§6)")
    say(f"Каталог данных: {data_dir}")
    say(f"Каталог результатов: {out_dir}")

    verify_code, verify_passed = _run_verifier(out_dir, say)
    if verify_code != 0:
        say("Подготовка остановлена: верификатор не прошёл.")
        _write_log(out_dir / "run_log.txt", log)
        return verify_code

    try:
        dataset = load_dataset(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        say(f"Подготовка остановлена: данные не читаются: {exc}")
        _write_log(out_dir / "run_log.txt", log)
        return 1

    problems = _contract_problems(dataset)
    reference_rows = _check_references(dataset, problems)
    input_rows = _check_inputs(dataset, problems)

    _write_csv(
        out_dir / "reference_check.csv",
        ["reference_id", "split", "split_label", "wave_class",
         "wave_class_label", "clean_exists", "clean_sha_ok", "samples",
         "annotation_exists", "annotation_sha_ok", "waves", "flagged_waves",
         "last_end_idx", "ok", "problems"],
        reference_rows)
    _write_csv(
        out_dir / "input_check.csv",
        ["input_id", "reference_id", "noise_repeat", "split", "split_label",
         "wave_class", "wave_class_label", "noisy_path", "exists",
         "sha256_ok", "samples", "finite", "ok", "problems"],
        input_rows)
    _write_csv(
        out_dir / "inputs.csv",
        ["input_id", "reference_id", "noise_repeat", "split", "split_label",
         "wave_class", "wave_class_label", "noisy_path"],
        [{key: row[key] for key in (
            "input_id", "reference_id", "noise_repeat", "split",
            "split_label", "wave_class", "wave_class_label", "noisy_path")}
         for row in input_rows])

    test_row = _choose_test_input(dataset)
    grid = filter_grid()
    elapsed = time.perf_counter() - started
    summary = _split_summary(dataset)
    python_cmd = _python_command()
    try:
        data_dir_label = str(data_dir.relative_to(ROOT))
    except ValueError:
        data_dir_label = str(data_dir)

    refs_ok = sum(1 for row in reference_rows if row["ok"])
    inputs_ok = sum(1 for row in input_rows if row["ok"])
    say(f"Эталоны: {refs_ok} из {len(reference_rows)} без замечаний")
    say(f"Шумовые входы: {inputs_ok} из {len(input_rows)} без замечаний")
    for split in SPLITS:
        item = summary[split]
        say(f"{SPLIT_LABELS[split]}: {item['references']} эталонов, "
            f"{item['inputs']} входов, классы {item['classes']}")
    say(f"Сетка фильтра: {len(grid)} сочетаний; окна среднего: "
        f"{list(MOVING_WINDOWS)}")
    say(f"Тестовый вход (§6): {test_row.input_id}")
    if problems:
        for problem in problems[:20]:
            say(f"[ОШИБКА ДАННЫХ] {problem}")
        if len(problems) > 20:
            say(f"[ОШИБКА ДАННЫХ] ... ещё {len(problems) - 20}")

    plan = {
        "stage": "prepare",
        "status": "ok" if not problems else "failed",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "dataset_version": DATASET_VERSION,
        "data_dir": data_dir_label,
        "fs_hz": 320.0,
        "samples_per_record": SAMPLES_PER_RECORD,
        "counts": {
            "references": len(dataset.references),
            "inputs": len(dataset.rows),
            "splits": summary,
        },
        "filters": {
            "grid": grid,
            "combinations": len(grid),
            "f_low_hz": list(FILTER_LOW_HZ),
            "f_high_hz": list(FILTER_HIGH_HZ),
            "orders": list(FILTER_ORDERS),
            "implementation": "butter(..., output='sos') + "
                              "sosfiltfilt(padtype='odd', padlen=None)",
        },
        "moving_average_windows": list(MOVING_WINDOWS),
        "test_input": {
            "input_id": test_row.input_id,
            "reference_id": test_row.reference_id,
            "wave_class": test_row.wave_class,
        },
        "stages": [
            {"name": "test", "status": "запланировано",
             "command": f"{python_cmd} run_experiment.py test"},
            {"name": "select", "status": "запланировано",
             "command": f"{python_cmd} run_experiment.py select"},
            {"name": "main", "status": "после коммита выбранных настроек",
             "command": f"{python_cmd} run_experiment.py main"},
        ],
        "verifier": {"exit_code": verify_code, "result": verify_passed},
        "integrity": {
            "references_ok": refs_ok,
            "references_total": len(reference_rows),
            "inputs_ok": inputs_ok,
            "inputs_total": len(input_rows),
            "problems": problems,
        },
        "environment": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "elapsed_sec": round(elapsed, 2),
    }
    (out_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    _write_summary(out_dir / "summary.md", plan, test_row.input_id,
                   verify_passed)
    say(f"План сохранён: {out_dir / 'plan.json'}")
    say(f"Длительность этапа: {elapsed:.2f} с")
    say("Итог: подготовка " + ("завершена." if not problems
                                else "остановлена из-за ошибок данных."))
    _write_log(out_dir / "run_log.txt", log)
    return 0 if not problems else 1


def _write_summary(path: Path, plan: dict, test_input_id: str,
                   verify_passed: str) -> None:
    counts = plan["counts"]
    lines = [
        "# Подготовка эксперимента",
        "",
        f"Статус: **{plan['status']}**. Сгенерировано {plan['generated_at']}; "
        f"коммит `{plan['git_commit']}`.",
        "",
        "Данные: `data/`, версия "
        f"`{plan['dataset_version']}`. Число сочетаний фильтра — "
        f"{plan['filters']['combinations']}.",
        "",
        "## Разбиение",
        "",
        "| Часть | Эталонов | Входов | clear | weak | absent |",
        "|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        item = counts["splits"][split]
        classes = item["classes"]
        lines.append(
            f"| {item['label']} | {item['references']} | {item['inputs']} | "
            f"{classes.get('clear', 0)} | {classes.get('weak', 0)} | "
            f"{classes.get('absent', 0)} |")
    integrity = plan["integrity"]
    lines += [
        "",
        "## Проверки",
        "",
        f"- Верификатор: код {plan['verifier']['exit_code']} "
        f"({verify_passed}).",
        f"- Эталоны без замечаний: {integrity['references_ok']} из "
        f"{integrity['references_total']}.",
        f"- Шумовые входы без замечаний: {integrity['inputs_ok']} из "
        f"{integrity['inputs_total']}.",
        f"- Тестовый вход §6: `{test_input_id}`.",
    ]
    if integrity["problems"]:
        lines.append("")
        lines.append("### Обнаруженные проблемы")
        lines += [f"- {problem}" for problem in integrity["problems"]]
    lines += [
        "",
        "## Сетка и этапы",
        "",
        f"- Нижние частоты: {plan['filters']['f_low_hz']} Гц.",
        f"- Верхние частоты: {plan['filters']['f_high_hz']} Гц.",
        f"- Порядки N: {plan['filters']['orders']}.",
        f"- Окна скользящего среднего: {plan['moving_average_windows']}.",
        "",
        "Следующий этап — `run_experiment.py test`; до экспериментальных "
        "прогонов план и код проверяются верификатором.",
        "",
        f"Длительность подготовки: {plan['elapsed_sec']} с.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_log(path: Path, log: list[str]) -> None:
    path.write_text("\n".join(log) + "\n", encoding="utf-8")


def _not_implemented(stage: str) -> int:
    print(f"Этап {stage!r} ещё не реализован. Сейчас доступен только "
          f"'prepare' (см. results/prepare/).")
    return 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Этапы эксперимента §6 методики.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="план расчётов, проверка данных и верификатора")
    prepare.add_argument("--data-dir", type=Path, default=ROOT / "data")
    prepare.add_argument("--results-dir", type=Path,
                         default=ROOT / "results")

    for name in ("test", "select", "main"):
        subparsers.add_parser(name, help=f"этап {name} (пока не реализован)")

    args = parser.parse_args(argv)
    if args.stage == "prepare":
        return run_prepare(args.data_dir.resolve(), args.results_dir.resolve())
    return _not_implemented(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
