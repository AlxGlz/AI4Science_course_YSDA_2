"""Этапы эксперимента §6 методики.

Реализованы этапы ``prepare`` (проверка данных и верификатора, план расчётов),
и ``test`` (однократный прогон на одном входе) и ``select`` (подбор
параметров на данных для подбора); ``main`` пока не реализован.

Команды (из корня проекта):

    Windows: ``.\\venv\\Scripts\\python.exe run_experiment.py prepare``
             ``.\\venv\\Scripts\\python.exe run_experiment.py test``
             ``.\\venv\\Scripts\\python.exe run_experiment.py select``
    macOS:   ``./.venv/bin/python run_experiment.py prepare``
             ``./.venv/bin/python run_experiment.py test``
             ``./.venv/bin/python run_experiment.py select``

Данные: ``data/manifest.csv``, ``data/references/``, ``data/noisy/``,
``data/annotations/``; чтение — через ``ppg.io``, вычисления — через
``ppg.experiment``, ``ppg.filters``, ``ppg.measure``, ``ppg.metrics``, ``ppg.methods``, ``ppg.selection``.
Проверка: ``verify.py`` (запускается этим скриптом первым шагом).
Используется в: этап ``main`` после фиксации настроек.
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

import numpy as np

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
from ppg.experiment import (  # noqa: E402
    EVAL_END_SEC,
    EVAL_START_SEC,
    compare_waves,
    error_series,
    etalon_refusal_summary,
    refusal_summary,
    select_evaluation_mask,
    value_counts,
)
from ppg.filters import butter_bandpass, moving_average  # noqa: E402
from ppg.measure import FS_DEFAULT  # noqa: E402
from ppg.methods import apply_method, method_settings  # noqa: E402
from ppg.selection import (  # noqa: E402
    choose_butterworth,
    choose_moving_average,
    count_admissible,
    is_admissible,
    peak_score,
)
from ppg.metrics import ABS_KEYS, nested_mean, refusal_stats  # noqa: E402

FILTER_LOW_HZ = (0.10, 0.30, 0.50)
FILTER_HIGH_HZ = (3.0, 5.0, 8.0, 12.0, 20.0)
FILTER_ORDERS = (2, 4)
MOVING_WINDOWS = (5, 11, 21, 41)
DATASET_VERSION = "vasotest-v2"
EXPECTED_REFERENCES = 48
EXPECTED_INPUTS = 240
EXPECTED_PER_SPLIT = 24
EXPECTED_PER_CLASS = 8
FS_HZ = 320.0

# Этап §6: тестовый вход, сырой сигнал, среднее с окном 11 и Баттерворт
# N=2 с полосой 0,30–8 Гц.
TEST_METHODS: tuple[dict[str, object], ...] = (
    {"name": "raw", "label": "сырой сигнал", "kind": "raw"},
    {"name": "moving_average_11",
     "label": "скользящее среднее, окно 11",
     "kind": "moving_average", "window": 11},
    {"name": "butterworth_0.30_8.0_N2",
     "label": "Баттерворт N=2, 0,30–8 Гц",
     "kind": "butterworth", "f_low_hz": 0.30, "f_high_hz": 8.0, "N": 2},
)


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


def _num(value: float | None, digits: int = 6) -> str:
    """Число для CSV; отказ (``None``) остаётся пустой ячейкой."""
    return "" if value is None else f"{value:.{digits}f}"


def _index(value: int | None) -> str:
    return "" if value is None else str(value)


def _summarize_comparisons(comparisons, input_id: str,
                           noise_repeat: int) -> dict[str, object]:
    """Ошибки и отказы метода: средние с равными весами и знаменатели (§2)."""
    absolute = error_series(comparisons)
    normalized = error_series(comparisons, normalized=True)
    return {
        "mean_absolute": {
            key: nested_mean({input_id: {noise_repeat: values}})
            for key, values in absolute.items()},
        "mean_normalized_pct": {
            key: nested_mean({input_id: {noise_repeat: values}})
            for key, values in normalized.items()},
        "values_count": value_counts(absolute),
        "refusals": refusal_summary(comparisons),
        "etalon_refusals": etalon_refusal_summary(comparisons),
    }


def run_test(data_dir: Path, results_dir: Path) -> int:
    """Этап ``test``: один вход, три обработки, ошибки и отказы (§6)."""
    started = time.perf_counter()
    out_dir = results_dir / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def say(message: str = "") -> None:
        print(message)
        log.append(message)

    say("run_experiment.py test — тестовый прогон на одном входе (§6)")
    say(f"Каталог данных: {data_dir}")
    say(f"Каталог результатов: {out_dir}")

    verify_code, verify_passed = _run_verifier(out_dir, say)
    if verify_code != 0:
        say("Тестовый прогон остановлен: верификатор не прошёл.")
        _write_log(out_dir / "run_log.txt", log)
        return verify_code

    try:
        dataset = load_dataset(data_dir)
        test_row = _choose_test_input(dataset)
    except (FileNotFoundError, ValueError) as exc:
        say(f"Тестовый прогон остановлен: данные не читаются: {exc}")
        _write_log(out_dir / "run_log.txt", log)
        return 1

    plan_path = results_dir / "prepare" / "plan.json"
    if plan_path.is_file():
        planned = json.loads(plan_path.read_text(encoding="utf-8"))
        planned_id = planned.get("test_input", {}).get("input_id")
        if planned_id and planned_id != test_row.input_id:
            say(f"Остановка: план подготовки указывает вход {planned_id}, "
                f"а правило §6 выбирает {test_row.input_id}.")
            _write_log(out_dir / "run_log.txt", log)
            return 1
        say(f"План подготовки подтверждает тестовый вход {test_row.input_id}.")

    reference = dataset.references[test_row.reference_id]
    clean_path = reference.clean_path
    noisy_path = (data_dir / test_row.noisy_path).resolve()
    clean_row = next(row for row in reference.inputs
                     if row.noise_repeat == 1)

    for label, path, expected_sha in (
            ("шумовой вход", noisy_path, test_row.noisy_sha256),
            ("чистый эталон", clean_path, clean_row.clean_sha256)):
        actual = sha256_file(path)
        if actual != expected_sha:
            say(f"Остановка: SHA-256 {label} не совпал с манифестом.")
            _write_log(out_dir / "run_log.txt", log)
            return 1

    try:
        signal = read_signal(noisy_path, expected_samples=SAMPLES_PER_RECORD)
        clean = read_signal(clean_path, expected_samples=SAMPLES_PER_RECORD)
    except ValueError as exc:
        say(f"Остановка: сигнал не читается: {exc}")
        _write_log(out_dir / "run_log.txt", log)
        return 1

    waves = reference.waves
    mask = select_evaluation_mask(waves, fs=FS_HZ)
    selected = int(np.count_nonzero(mask))
    say(f"Вход {test_row.input_id}: эталон {test_row.reference_id}, "
        f"часть «{SPLIT_LABELS[test_row.split]}», класс "
        f"«{CLASS_LABELS[test_row.wave_class]}», реализация "
        f"№{test_row.noise_repeat}.")
    say(f"Волн в разметке: {len(waves)}; в окне оценки "
        f"{EVAL_START_SEC:.0f}–{EVAL_END_SEC:.0f} с: {selected} "
        f"(с вторым пиком {int(np.count_nonzero(waves.has_diastolic_peak[mask]))}).")

    method_results: list[dict[str, object]] = []
    measurement_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    refusal_rows: list[dict[str, object]] = []
    try:
        for method in TEST_METHODS:
            processed = apply_method(signal, method, fs=FS_HZ)
            comparisons = compare_waves(waves, clean, processed, mask, fs=FS_HZ)
            summary = _summarize_comparisons(
                comparisons, test_row.input_id, test_row.noise_repeat)
            say(f"{method['label']}: ошибки A1 {_num(summary['mean_normalized_pct']['A1'], 3)} %, "
                f"t1 {_num(summary['mean_normalized_pct']['t1'], 3)} %, "
                f"A2 {_num(summary['mean_normalized_pct']['A2'], 3)} %, "
                f"t2 {_num(summary['mean_normalized_pct']['t2'], 3)} %; "
                f"отказы 1-го пика "
                f"{summary['refusals']['first_peak']['expected'] - summary['refusals']['first_peak']['successes']}, "
                f"2-го пика "
                f"{summary['refusals']['second_peak']['expected'] - summary['refusals']['second_peak']['successes']}.")

            for item in comparisons:
                row = {
                    "method": method["name"],
                    "method_label": method["label"],
                    "wave_index": item.wave_index,
                    "start_idx": item.start_idx,
                    "end_idx": item.end_idx,
                    "has_diastolic_peak": int(item.has_diastolic_peak),
                }
                for prefix, measurement in (("etalon", item.etalon),
                                            ("estimate", item.estimate)):
                    row.update({
                        f"{prefix}_p1": _index(measurement.p1),
                        f"{prefix}_A1": _num(measurement.A1),
                        f"{prefix}_t1": _num(measurement.t1),
                        f"{prefix}_p2": _index(measurement.p2),
                        f"{prefix}_A2": _num(measurement.A2),
                        f"{prefix}_t2": _num(measurement.t2),
                        f"{prefix}_T": _num(measurement.T),
                    })
                row.update({
                    "refused_first": int(item.estimate.refused_first),
                    "refused_second": int(item.estimate.refused_second),
                    "second_required": int(item.estimate.second_required),
                })
                for key in ABS_KEYS:
                    row[f"abs_{key}"] = _num(item.absolute[key])
                for key in ABS_KEYS:
                    row[f"norm_{key}"] = _num(item.normalized[key])
                measurement_rows.append(row)

            for key in ABS_KEYS:
                error_rows.append({
                    "method": method["name"],
                    "method_label": method["label"],
                    "metric": key,
                    "mean_absolute": _num(summary["mean_absolute"][key]),
                    "mean_normalized_pct": _num(
                        summary["mean_normalized_pct"][key]),
                    "values_count": summary["values_count"][key],
                })

            for source, stats in (("обработка", summary["refusals"]),
                                  ("эталон", summary["etalon_refusals"])):
                for peak, item in stats.items():
                    refusal_rows.append({
                        "source": source,
                        "method": method["name"],
                        "peak": peak,
                        "expected": item["expected"],
                        "successes": item["successes"],
                        "refusals": item["expected"] - item["successes"],
                        "refusal_pct": _num(item["refusal_pct"], 3),
                    })

            method_results.append({
                "name": method["name"],
                "label": method["label"],
                "settings": method_settings(method),
                "mean_absolute": summary["mean_absolute"],
                "mean_normalized_pct": summary["mean_normalized_pct"],
                "values_count": summary["values_count"],
                "refusals": summary["refusals"],
                "etalon_refusals": summary["etalon_refusals"],
            })
    except ValueError as exc:
        say(f"Остановка: расчёт прерван ({exc}). §6 требует исправить "
            f"программу до продолжения.")
        _write_log(out_dir / "run_log.txt", log)
        return 1

    measurement_columns = [
        "method", "method_label", "wave_index", "start_idx", "end_idx",
        "has_diastolic_peak",
        "etalon_p1", "etalon_A1", "etalon_t1", "etalon_p2", "etalon_A2",
        "etalon_t2", "etalon_T",
        "estimate_p1", "estimate_A1", "estimate_t1", "estimate_p2",
        "estimate_A2", "estimate_t2", "estimate_T",
        "refused_first", "refused_second", "second_required",
        "abs_A1", "abs_t1", "abs_A2", "abs_t2",
        "norm_A1", "norm_t1", "norm_A2", "norm_t2",
    ]
    _write_csv(out_dir / "measurements.csv", measurement_columns,
               measurement_rows)
    _write_csv(
        out_dir / "errors.csv",
        ["method", "method_label", "metric", "mean_absolute",
         "mean_normalized_pct", "values_count"],
        error_rows)
    _write_csv(
        out_dir / "refusals.csv",
        ["source", "method", "peak", "expected", "successes", "refusals",
         "refusal_pct"],
        refusal_rows)

    elapsed = time.perf_counter() - started
    try:
        data_dir_label = str(data_dir.relative_to(ROOT))
    except ValueError:
        data_dir_label = str(data_dir)
    report = {
        "stage": "test",
        "status": "ok",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "dataset_version": DATASET_VERSION,
        "data_dir": data_dir_label,
        "fs_hz": FS_HZ,
        "evaluation_window_sec": [EVAL_START_SEC, EVAL_END_SEC],
        "test_input": {
            "input_id": test_row.input_id,
            "reference_id": test_row.reference_id,
            "split": test_row.split,
            "wave_class": test_row.wave_class,
            "noise_repeat": test_row.noise_repeat,
            "noisy_path": test_row.noisy_path,
            "noisy_sha256": test_row.noisy_sha256,
            "clean_path": clean_path.relative_to(data_dir).as_posix(),
            "clean_sha256": clean_row.clean_sha256,
            "annotation_sha256": test_row.annotation_sha256,
        },
        "waves": {
            "total": len(waves),
            "evaluated": selected,
            "evaluated_with_second_peak": int(
                np.count_nonzero(waves.has_diastolic_peak[mask])),
        },
        "methods": method_results,
        "verifier": {"exit_code": verify_code, "result": verify_passed},
        "environment": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "elapsed_sec": round(elapsed, 2),
    }
    (out_dir / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    _write_test_summary(out_dir / "summary.md", report)
    say(f"Измерения: {out_dir / 'measurements.csv'}")
    say(f"Длительность этапа: {elapsed:.2f} с")
    say("Итог: тестовый прогон завершён; отказы и ошибки записаны явно.")
    _write_log(out_dir / "run_log.txt", log)
    return 0


def _write_test_summary(path: Path, report: dict) -> None:
    test = report["test_input"]
    waves = report["waves"]
    lines = [
        "# Тестовый прогон",
        "",
        f"Статус: **{report['status']}**. Сгенерировано "
        f"{report['generated_at']}; коммит `{report['git_commit']}`.",
        "",
        f"Вход `{test['input_id']}`: эталон `{test['reference_id']}`, часть "
        f"«{SPLIT_LABELS[test['split']]}», класс "
        f"«{CLASS_LABELS[test['wave_class']]}», реализация "
        f"№{test['noise_repeat']}.",
        f"Окно оценки §2: {report['evaluation_window_sec'][0]:.0f}–"
        f"{report['evaluation_window_sec'][1]:.0f} с; волн в разметке "
        f"{waves['total']}, оценено {waves['evaluated']}, из них с вторым "
        f"пиком {waves['evaluated_with_second_peak']}.",
        "",
        "## Нормированные ошибки (%, среднее по волнам)",
        "",
        "| Обработка | E(A1) | E(t1) | E(A2) | E(t2) |",
        "|---|---|---|---|---|",
    ]
    for method in report["methods"]:
        values = method["mean_normalized_pct"]
        lines.append(
            f"| {method['label']} | {_num(values['A1'], 3)} | "
            f"{_num(values['t1'], 3)} | {_num(values['A2'], 3)} | "
            f"{_num(values['t2'], 3)} |")
    lines += [
        "",
        "## Абсолютные ошибки (условные единицы и мс)",
        "",
        "| Обработка | A1 | t1 | A2 | t2 |",
        "|---|---|---|---|---|",
    ]
    for method in report["methods"]:
        values = method["mean_absolute"]
        lines.append(
            f"| {method['label']} | {_num(values['A1'], 4)} | "
            f"{_num(values['t1'], 3)} | {_num(values['A2'], 4)} | "
            f"{_num(values['t2'], 3)} |")
    lines += [
        "",
        "## Отказы измерителя",
        "",
        "| Обработка | Пик | Ожидалось | Успешно | Отказов | Доля, % |",
        "|---|---|---|---|---|---|",
    ]
    for method in report["methods"]:
        for peak, label in (("first_peak", "первый"),
                            ("second_peak", "второй")):
            item = method["refusals"][peak]
            share = ("не применимо" if item["refusal_pct"] is None
                     else _num(item["refusal_pct"], 3))
            lines.append(
                f"| {method['label']} | {label} | {item['expected']} | "
                f"{item['successes']} | "
                f"{item['expected'] - item['successes']} | {share} |")
    etalon = report["methods"][0]["etalon_refusals"]
    lines += [
        "",
        f"Отказы на чистом эталоне: первый пик — "
        f"{etalon['first_peak']['expected'] - etalon['first_peak']['successes']} "
        f"из {etalon['first_peak']['expected']}, второй — "
        f"{etalon['second_peak']['expected'] - etalon['second_peak']['successes']} "
        f"из {etalon['second_peak']['expected']}.",
        "",
        "## Графики",
        "",
        "Обзор записи и увеличенный фрагмент с началами волн и найденными "
        "пиками строятся по сохранённым настройкам командой "
        "`plot_comparison.py --results results/test`; оглавление — "
        "`figures/index.md`.",
        "",
        "Знаменатели средних и полные измерения по волнам — в "
        "`measurements.csv` и `errors.csv`. Превосходство фильтра в тестовом "
        "прогоне не требуется (§6); результаты не входят в итоговое "
        "статистическое сравнение.",
        "",
        f"Верификатор: код {report['verifier']['exit_code']} "
        f"({report['verifier']['result']}).",
        f"Длительность этапа: {report['elapsed_sec']} с.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Русская форма существительного после числа."""
    if count % 10 == 1 and count % 100 != 11:
        return one
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


def _cell(value: float | None, digits: int = 3) -> str:
    """Число для markdown-таблицы; не вычисленное значение — прочерк."""
    return "—" if value is None else f"{value:.{digits}f}"


SELECT_SPLIT = "tune"


def select_methods() -> list[dict[str, object]]:
    """Варианты подбора §6: сырой сигнал, 4 окна среднего, 30 Баттервортов."""
    methods: list[dict[str, object]] = [
        {"name": "raw", "label": "сырой сигнал", "kind": "raw"}]
    for window in MOVING_WINDOWS:
        methods.append({
            "name": f"moving_average_{window}",
            "label": f"скользящее среднее, окно {window}",
            "kind": "moving_average",
            "window": window,
        })
    for combination in filter_grid():
        f_low = float(combination["f_low_hz"])
        f_high = float(combination["f_high_hz"])
        order = int(combination["N"])
        label = f"Баттерворт N={order}, {f_low:.2f}–{f_high:g} Гц"
        methods.append({
            "name": f"butterworth_{f_low:.2f}_{f_high:.1f}_N{order}",
            "label": label.replace(".", ","),
            "kind": "butterworth",
            **combination,
        })
    return methods


def _ru_float(value: float, digits: int = 2) -> str:
    """Число с десятичной запятой для подписей на русском."""
    return f"{float(value):.{digits}f}".replace(".", ",")


def settings_label(settings: dict[str, object]) -> str:
    """Короткое описание параметров обработки для таблиц."""
    kind = settings.get("kind")
    if kind == "raw":
        return "—"
    if kind == "moving_average":
        return f"окно {settings['window']}"
    if kind == "butterworth":
        return (f"N={settings['N']}, "
                f"{_ru_float(settings['f_low_hz'])}–"
                f"{float(settings['f_high_hz']):g} Гц")
    return str(kind)


def _check_sha(path: Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 {label} не совпал с манифестом")


def _load_select_inputs(dataset) -> dict[str, dict[str, object]]:
    """24 эталона части tune и их 120 шумовых входов с проверкой SHA (§6)."""
    rows = [row for row in dataset.rows if row.split == SELECT_SPLIT]
    reference_ids = sorted({row.reference_id for row in rows}, key=natural_key)
    expected_inputs = EXPECTED_PER_SPLIT * 5
    if len(reference_ids) != EXPECTED_PER_SPLIT or len(rows) != expected_inputs:
        raise ValueError(
            f"в части {SELECT_SPLIT}: эталонов {len(reference_ids)} "
            f"(ожидалось {EXPECTED_PER_SPLIT}), входов {len(rows)} "
            f"(ожидалось {expected_inputs})")
    loaded: dict[str, dict[str, object]] = {}
    for reference_id in reference_ids:
        reference = dataset.references[reference_id]
        if any(row.split != SELECT_SPLIT for row in reference.inputs):
            raise ValueError(f"эталон {reference_id}: есть входы вне части "
                             f"{SELECT_SPLIT}")
        _check_sha(reference.clean_path, reference.inputs[0].clean_sha256,
                   f"чистого эталона {reference_id}")
        inputs = sorted(reference.inputs, key=lambda row: row.noise_repeat)
        signals: dict[str, object] = {}
        for row in inputs:
            path = (dataset.data_dir / row.noisy_path).resolve()
            _check_sha(path, row.noisy_sha256, f"входа {row.input_id}")
            signals[row.input_id] = read_signal(
                path, expected_samples=SAMPLES_PER_RECORD)
        mask = select_evaluation_mask(reference.waves, fs=FS_HZ)
        if not mask.any():
            raise ValueError(
                f"эталон {reference_id}: в окне "
                f"{EVAL_START_SEC:.0f}–{EVAL_END_SEC:.0f} с нет полных волн")
        loaded[reference_id] = {
            "rows": inputs,
            "clean": reference.clean,
            "waves": reference.waves,
            "mask": mask,
            "signals": signals,
        }
    return loaded


def _nested_error(groups, key: str) -> float | None:
    return nested_mean({
        reference_id: {input_id: values[key]
                       for input_id, values in inputs.items()}
        for reference_id, inputs in groups.items()})


def _error_count(groups, key: str) -> int:
    return sum(1 for inputs in groups.values()
               for values in inputs.values()
               for value in values[key] if value is not None)


def _select_summary(method: dict[str, object], norm_groups, abs_groups,
                    totals) -> dict[str, object]:
    """Ошибки и отказы варианта: средние §2 по 24 эталонам и 5 повторам."""
    return {
        "name": method["name"],
        "label": method["label"],
        "settings": method_settings(method),
        "mean_absolute": {key: _nested_error(abs_groups, key)
                          for key in ABS_KEYS},
        "mean_normalized_pct": {key: _nested_error(norm_groups, key)
                                for key in ABS_KEYS},
        "values_count": {key: _error_count(abs_groups, key)
                         for key in ABS_KEYS},
        "refusals": {
            "first_peak": refusal_stats(*totals["first"]),
            "second_peak": refusal_stats(*totals["second"]),
        },
    }


def _brief(entry: dict[str, object] | None) -> dict[str, object] | None:
    """Краткая запись выбранного варианта для ``run.json``."""
    if entry is None:
        return None
    return {
        "name": entry["name"],
        "label": entry["label"],
        "settings": entry["settings"],
        "score_pct": entry["score_pct"],
        "mean_absolute": entry["mean_absolute"],
        "mean_normalized_pct": entry["mean_normalized_pct"],
    }


def run_select(data_dir: Path, results_dir: Path) -> int:
    """Этап ``select``: 35 вариантов на данных tune и выбор по §7 (§6)."""
    started = time.perf_counter()
    out_dir = results_dir / "select"
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def say(message: str = "") -> None:
        print(message)
        log.append(message)

    say("run_experiment.py select — подбор параметров на данных tune (§6, §7)")
    say(f"Каталог данных: {data_dir}")
    say(f"Каталог результатов: {out_dir}")

    verify_code, verify_passed = _run_verifier(out_dir, say)
    if verify_code != 0:
        say("Подбор остановлен: верификатор не прошёл.")
        _write_log(out_dir / "run_log.txt", log)
        return verify_code

    try:
        dataset = load_dataset(data_dir)
        loaded = _load_select_inputs(dataset)
    except (FileNotFoundError, ValueError) as exc:
        say(f"Подбор остановлен: данные не готовы: {exc}")
        _write_log(out_dir / "run_log.txt", log)
        return 1

    reference_ids = list(loaded)
    methods = select_methods()
    waves_total = sum(len(item["waves"]) for item in loaded.values())
    waves_evaluated = sum(int(item["mask"].sum()) for item in loaded.values())
    waves_second = sum(
        int(item["waves"].has_diastolic_peak[item["mask"]].sum())
        for item in loaded.values())
    inputs_total = sum(len(item["rows"]) for item in loaded.values())
    say(f"Эталонов: {len(reference_ids)}, шумовых входов: {inputs_total}; "
        f"вариантов обработки: {len(methods)}.")
    say(f"Волн: всего {waves_total}, в окне оценки {waves_evaluated}, "
        f"с вторым пиком {waves_second}.")

    entries: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    refusal_rows: list[dict[str, object]] = []
    by_reference_rows: list[dict[str, object]] = []
    etalon_totals = {"first": [0, 0], "second": [0, 0]}
    try:
        for index, method in enumerate(methods):
            method_started = time.perf_counter()
            norm_groups: dict[str, dict[str, dict[str, list]]] = {}
            abs_groups: dict[str, dict[str, dict[str, list]]] = {}
            totals = {"first": [0, 0], "second": [0, 0]}
            for reference_id, item in loaded.items():
                norm_ref = norm_groups.setdefault(reference_id, {})
                abs_ref = abs_groups.setdefault(reference_id, {})
                for row in item["rows"]:
                    processed = apply_method(item["signals"][row.input_id],
                                             method, fs=FS_HZ)
                    comparisons = compare_waves(
                        item["waves"], item["clean"], processed,
                        item["mask"], fs=FS_HZ)
                    normalized = error_series(comparisons, normalized=True)
                    absolute = error_series(comparisons)
                    norm_input = norm_ref.setdefault(row.input_id, {})
                    abs_input = abs_ref.setdefault(row.input_id, {})
                    for key in ABS_KEYS:
                        norm_input.setdefault(key, []).extend(normalized[key])
                        abs_input.setdefault(key, []).extend(absolute[key])
                    refusals = refusal_summary(comparisons)
                    for peak in ("first", "second"):
                        stats = refusals[f"{peak}_peak"]
                        totals[peak][0] += stats["expected"]
                        totals[peak][1] += stats["successes"]
                    if index == 0:
                        reference_stats = etalon_refusal_summary(comparisons)
                        for peak in ("first", "second"):
                            stats = reference_stats[f"{peak}_peak"]
                            etalon_totals[peak][0] += stats["expected"]
                            etalon_totals[peak][1] += stats["successes"]

            entry = _select_summary(method, norm_groups, abs_groups, totals)
            entry["score_pct"] = peak_score(entry)
            entry["admissible"] = is_admissible(entry)
            entry["elapsed_sec"] = round(
                time.perf_counter() - method_started, 2)
            entries.append(entry)

            values = entry["mean_normalized_pct"]
            first_refusals = (entry["refusals"]["first_peak"]["expected"]
                              - entry["refusals"]["first_peak"]["successes"])
            second_refusals = (entry["refusals"]["second_peak"]["expected"]
                               - entry["refusals"]["second_peak"]["successes"])
            verdict = "допустим" if entry["admissible"] else "не допустим"
            say(f"[{index + 1:2d}/{len(methods)}] {entry['label']}: "
                f"E_A1 {_cell(values['A1'])} %, E_t1 {_cell(values['t1'])} %, "
                f"E_A2 {_cell(values['A2'])} %, "
                f"E_t2 {_cell(values['t2'])} %; "
                f"отказы {first_refusals}/{second_refusals}; "
                f"оценка {_cell(entry['score_pct'])}; {verdict}")

            for key in ABS_KEYS:
                error_rows.append({
                    "method": entry["name"],
                    "method_label": entry["label"],
                    "kind": entry["settings"].get("kind"),
                    "metric": key,
                    "mean_absolute": _num(entry["mean_absolute"][key]),
                    "mean_normalized_pct": _num(
                        entry["mean_normalized_pct"][key]),
                    "values_count": entry["values_count"][key],
                })
            for peak, peak_label in (("first_peak", "первый"),
                                     ("second_peak", "второй")):
                stats = entry["refusals"][peak]
                refusal_rows.append({
                    "method": entry["name"],
                    "method_label": entry["label"],
                    "kind": entry["settings"].get("kind"),
                    "peak": peak_label,
                    "expected": stats["expected"],
                    "successes": stats["successes"],
                    "refusals": stats["expected"] - stats["successes"],
                    "refusal_pct": _num(stats["refusal_pct"]),
                })
            for reference_id, inputs in norm_groups.items():
                reference = dataset.references[reference_id]
                abs_inputs = abs_groups[reference_id]
                for key in ABS_KEYS:
                    norm_value = nested_mean({reference_id: {
                        input_id: values[key]
                        for input_id, values in inputs.items()}})
                    abs_value = nested_mean({reference_id: {
                        input_id: values[key]
                        for input_id, values in abs_inputs.items()}})
                    by_reference_rows.append({
                        "method": entry["name"],
                        "method_label": entry["label"],
                        "reference_id": reference_id,
                        "wave_class": reference.wave_class,
                        "wave_class_label": CLASS_LABELS[
                            reference.wave_class],
                        "metric": key,
                        "mean_absolute": _num(abs_value),
                        "mean_normalized_pct": _num(norm_value),
                        "values_count": _error_count({reference_id: inputs},
                                                     key),
                    })
    except ValueError as exc:
        say(f"Остановка: расчёт прерван ({exc}). §6 требует исправить "
            f"программу до продолжения.")
        _write_log(out_dir / "run_log.txt", log)
        return 1

    butterworth = [entry for entry in entries
                   if entry["settings"].get("kind") == "butterworth"]
    moving = [entry for entry in entries
              if entry["settings"].get("kind") == "moving_average"]
    raw_entry = next(entry for entry in entries
                     if entry["settings"].get("kind") == "raw")
    chosen_butterworth = choose_butterworth(butterworth)
    chosen_moving = choose_moving_average(moving)
    admissible_counts = {
        "butterworth": count_admissible(butterworth),
        "moving_average": count_admissible(moving),
    }

    if chosen_butterworth is None:
        termination = {
            "butterworth_found": False,
            "moving_average_found": chosen_moving is not None,
            "action": ("Допустимый Баттерворт не найден: основной прогон "
                       "не выполняется, таблица подбора сохранена (§7.7)"),
        }
    elif chosen_moving is None:
        termination = {
            "butterworth_found": True,
            "moving_average_found": False,
            "action": ("Допустимое среднее не найдено: основной прогон "
                       "сравнит Баттерворт только с сырым сигналом; полное "
                       "подтверждение гипотезы невозможно (§7.7)"),
        }
    else:
        termination = {
            "butterworth_found": True,
            "moving_average_found": True,
            "action": ("Настройки выбраны; основной прогон — после фиксации "
                       "настроек коммитом (§7.4)"),
        }

    stage_methods = [raw_entry]
    if chosen_moving is not None:
        stage_methods.append(chosen_moving)
    if chosen_butterworth is not None:
        stage_methods.append(chosen_butterworth)

    selection = {
        "rule": "§7",
        "thresholds": {
            "e_a1_max_pct": 5.0,
            "e_t1_max_pct": 5.0,
            "refusals": "не допускаются ни на одном пике",
        },
        "score": "минимум max(E_A2, E_t2) после усреднения §2",
        "tie_break": {
            "moving_average": "меньшее окно",
            "butterworth": ("меньший N, затем меньшая нижняя, затем "
                            "меньшая верхняя частота"),
        },
        "admissible_counts": admissible_counts,
        "chosen": {
            "butterworth": _brief(chosen_butterworth),
            "moving_average": _brief(chosen_moving),
            "raw": _brief(raw_entry),
        },
        "termination": termination,
    }

    evaluated_inputs = sorted(
        (row.input_id for item in loaded.values() for row in item["rows"]),
        key=natural_key)
    candidates_rows = []
    for entry in entries:
        settings = entry["settings"]
        refusals = entry["refusals"]
        candidates_rows.append({
            "method": entry["name"],
            "method_label": entry["label"],
            "kind": settings.get("kind"),
            "window": _index(settings.get("window")),
            "f_low_hz": _num(settings.get("f_low_hz")),
            "f_high_hz": _num(settings.get("f_high_hz")),
            "N": _index(settings.get("N")),
            "e_a1_pct": _num(entry["mean_normalized_pct"]["A1"]),
            "e_t1_pct": _num(entry["mean_normalized_pct"]["t1"]),
            "e_a2_pct": _num(entry["mean_normalized_pct"]["A2"]),
            "e_t2_pct": _num(entry["mean_normalized_pct"]["t2"]),
            "score_pct": _num(entry["score_pct"]),
            "admissible": int(entry["admissible"]),
            "refusals_first": refusals["first_peak"]["expected"]
                              - refusals["first_peak"]["successes"],
            "refusals_second": refusals["second_peak"]["expected"]
                               - refusals["second_peak"]["successes"],
            "elapsed_sec": entry["elapsed_sec"],
        })

    _write_csv(
        out_dir / "errors.csv",
        ["method", "method_label", "kind", "metric", "mean_absolute",
         "mean_normalized_pct", "values_count"],
        error_rows)
    _write_csv(
        out_dir / "refusals.csv",
        ["method", "method_label", "kind", "peak", "expected", "successes",
         "refusals", "refusal_pct"],
        refusal_rows)
    _write_csv(
        out_dir / "by_reference.csv",
        ["method", "method_label", "reference_id", "wave_class",
         "wave_class_label", "metric", "mean_absolute", "mean_normalized_pct",
         "values_count"],
        by_reference_rows)
    _write_csv(
        out_dir / "candidates.csv",
        ["method", "method_label", "kind", "window", "f_low_hz", "f_high_hz",
         "N", "e_a1_pct", "e_t1_pct", "e_a2_pct", "e_t2_pct", "score_pct",
         "admissible", "refusals_first", "refusals_second", "elapsed_sec"],
        candidates_rows)

    elapsed = time.perf_counter() - started
    try:
        data_dir_label = str(data_dir.relative_to(ROOT))
    except ValueError:
        data_dir_label = str(data_dir)
    report = {
        "stage": "select",
        "status": "ok",
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "git_commit": _git_commit(),
        "dataset_version": DATASET_VERSION,
        "data_dir": data_dir_label,
        "fs_hz": FS_HZ,
        "evaluation_window_sec": [EVAL_START_SEC, EVAL_END_SEC],
        "split": SELECT_SPLIT,
        "counts": {
            "references": len(reference_ids),
            "inputs": inputs_total,
            "methods": len(methods),
            "butterworth": len(butterworth),
            "moving_average": len(moving),
            "raw": 1,
        },
        "waves": {
            "total": waves_total,
            "evaluated": waves_evaluated,
            "evaluated_with_second_peak": waves_second,
        },
        "etalon_refusals": {
            "first_peak": refusal_stats(*etalon_totals["first"]),
            "second_peak": refusal_stats(*etalon_totals["second"]),
        },
        "methods": stage_methods,
        "candidates": entries,
        "selection": selection,
        "evaluated_inputs": evaluated_inputs,
        "verifier": {"exit_code": verify_code, "result": verify_passed},
        "environment": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "elapsed_sec": round(elapsed, 2),
    }
    (out_dir / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    _write_select_summary(out_dir / "summary.md", report)

    say("")
    say("Выбранные настройки (§7):")
    for title, chosen in (("Баттерворт", chosen_butterworth),
                          ("среднее", chosen_moving)):
        if chosen is None:
            say(f"  {title}: допустимого варианта нет")
        else:
            say(f"  {title}: {chosen['label']} — оценка "
                f"{_cell(chosen['score_pct'])} %")
    say(f"Итог §7.7: {termination['action']}.")
    if chosen_butterworth is not None:
        say("Следующий шаг: зафиксировать настройки коммитом до main.")
    say(f"Длительность этапа: {elapsed:.2f} с")
    _write_log(out_dir / "run_log.txt", log)
    return 0


def _write_select_summary(path: Path, report: dict) -> None:
    selection = report["selection"]
    counts = report["counts"]
    waves = report["waves"]
    window = report["evaluation_window_sec"]
    lines = [
        "# Подбор параметров",
        "",
        f"Статус: **{report['status']}**. Сгенерировано "
        f"{report['generated_at']}; коммит `{report['git_commit']}`.",
        "",
        f"Данные для подбора (§6): {counts['references']} "
        f"{_plural(counts['references'], 'эталон', 'эталона', 'эталонов')}, "
        f"{counts['inputs']} шумовых "
        f"{_plural(counts['inputs'], 'вход', 'входа', 'входов')}; "
        f"вариантов обработки "
        f"{counts['methods']}: сырой сигнал, {counts['moving_average']} "
        f"{_plural(counts['moving_average'], 'окно', 'окна', 'окон')} "
        f"среднего, {counts['butterworth']} "
        f"{_plural(counts['butterworth'], 'полоса', 'полосы', 'полос')} "
        f"Баттерворта.",
        f"Волн в окне оценки {window[0]:.0f}–{window[1]:.0f} с: "
        f"{waves['evaluated']} из {waves['total']}, с вторым пиком "
        f"{waves['evaluated_with_second_peak']}.",
        "",
        "## Правило §7",
        "",
        f"- Допустимы варианты без отказов и с E(A1) ≤ "
        f"{selection['thresholds']['e_a1_max_pct']:g} %, E(t1) ≤ "
        f"{selection['thresholds']['e_t1_max_pct']:g} %.",
        f"- Оценка: {selection['score']}.",
        "- Равенство оценок: среднее — меньшее окно; Баттерворт — меньший N, "
        "затем меньшая нижняя, затем меньшая верхняя частота.",
        "",
        "## Выбранные настройки",
        "",
        "| Обработка | Параметры | E(A1), % | E(t1), % | E(A2), % | "
        "E(t2), % | Оценка, % |",
        "|---|---|---|---|---|---|---|",
    ]
    for title, key in (("Баттерворт", "butterworth"),
                       ("Скользящее среднее", "moving_average")):
        chosen = selection["chosen"][key]
        if chosen is None:
            lines.append(f"| {title} | нет допустимого варианта | | | | | |")
            continue
        values = chosen["mean_normalized_pct"]
        lines.append(
            f"| {title} | {settings_label(chosen['settings'])} | "
            f"{_cell(values['A1'])} | {_cell(values['t1'])} | "
            f"{_cell(values['A2'])} | {_cell(values['t2'])} | "
            f"{_cell(chosen['score_pct'])} |")
    raw = selection["chosen"]["raw"]
    values = raw["mean_normalized_pct"]
    lines += [
        f"| Сырой сигнал (база) | — | {_cell(values['A1'])} | "
        f"{_cell(values['t1'])} | {_cell(values['A2'])} | "
        f"{_cell(values['t2'])} | {_cell(raw['score_pct'])} |",
        "",
        "## Абсолютные ошибки выбранных вариантов",
        "",
        "| Обработка | A1 | t1 | A2 | t2 |",
        "|---|---|---|---|---|",
    ]
    for title, key in (("Баттерворт", "butterworth"),
                       ("Скользящее среднее", "moving_average"),
                       ("Сырой сигнал", "raw")):
        chosen = selection["chosen"][key]
        if chosen is None:
            continue
        values = chosen["mean_absolute"]
        lines.append(
            f"| {title} | {_cell(values['A1'], 4)} | "
            f"{_cell(values['t1'], 3)} | {_cell(values['A2'], 4)} | "
            f"{_cell(values['t2'], 3)} |")
    lines += [
        "",
        "Амплитуды — условные единицы, времена — мс.",
        "",
        "## Решение по §7.7",
        "",
        selection["termination"]["action"] + ".",
        "",
        f"Допустимых вариантов: Баттерворт — "
        f"{selection['admissible_counts']['butterworth']} из "
        f"{counts['butterworth']}, среднее — "
        f"{selection['admissible_counts']['moving_average']} из "
        f"{counts['moving_average']}.",
        "",
    ]

    butterworth = [entry for entry in report["candidates"]
                   if entry["settings"].get("kind") == "butterworth"]
    moving = [entry for entry in report["candidates"]
              if entry["settings"].get("kind") == "moving_average"]

    def order(entry):
        score = entry["score_pct"]
        settings = entry["settings"]
        return (not entry["admissible"],
                float("inf") if score is None else score,
                int(settings.get("N") or 0),
                float(settings.get("f_low_hz") or 0.0),
                float(settings.get("f_high_hz") or 0.0),
                int(settings.get("window") or 0))

    lines += [
        "## Варианты Баттерворта (допустимые сверху)",
        "",
        "| N | Нижняя, Гц | Верхняя, Гц | E(A1), % | E(t1), % | E(A2), % | "
        "E(t2), % | Оценка, % | Допустим |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in sorted(butterworth, key=order):
        settings = entry["settings"]
        values = entry["mean_normalized_pct"]
        lines.append(
            f"| {settings['N']} | {float(settings['f_low_hz']):.2f} | "
            f"{float(settings['f_high_hz']):g} | "
            f"{_cell(values['A1'])} | {_cell(values['t1'])} | "
            f"{_cell(values['A2'])} | {_cell(values['t2'])} | "
            f"{_cell(entry['score_pct'])} | "
            f"{'да' if entry['admissible'] else 'нет'} |")

    lines += [
        "",
        "## Скользящее среднее",
        "",
        "| Окно | E(A1), % | E(t1), % | E(A2), % | E(t2), % | Оценка, % | "
        "Допустим |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in sorted(moving, key=order):
        values = entry["mean_normalized_pct"]
        lines.append(
            f"| {entry['settings']['window']} | {_cell(values['A1'])} | "
            f"{_cell(values['t1'])} | {_cell(values['A2'])} | "
            f"{_cell(values['t2'])} | {_cell(entry['score_pct'])} | "
            f"{'да' if entry['admissible'] else 'нет'} |")

    lines += [
        "",
        "## Диагностика",
        "",
        "- Полные ошибки по вариантам: `errors.csv`; отказы: `refusals.csv`; "
        "сводка по эталонам: `by_reference.csv`; параметры и допуск "
        "вариантов: `candidates.csv`.",
        "- Выбранные настройки сохранены в `run.json` (`methods`, "
        "`selection`) — их читают основной прогон и графики.",
        "- Графики по сохранённым настройкам: "
        "`plot_comparison.py --results results/select`; другие варианты — "
        "параметром `--method`.",
        "",
        f"Верификатор: код {report['verifier']['exit_code']} "
        f"({report['verifier']['result']}).",
        f"Длительность этапа: {report['elapsed_sec']} с.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _not_implemented(stage: str) -> int:
    print(f"Этап {stage!r} ещё не реализован. Доступны этапы "
          f"'prepare', 'test' и 'select' (см. results/).")
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

    test = subparsers.add_parser(
        "test", help="тестовый прогон на одном входе (§6)")
    test.add_argument("--data-dir", type=Path, default=ROOT / "data")
    test.add_argument("--results-dir", type=Path, default=ROOT / "results")

    select = subparsers.add_parser(
        "select", help="подбор параметров на данных для подбора (§6)")
    select.add_argument("--data-dir", type=Path, default=ROOT / "data")
    select.add_argument("--results-dir", type=Path, default=ROOT / "results")

    subparsers.add_parser("main", help="этап main (пока не реализован)")

    args = parser.parse_args(argv)
    if args.stage == "prepare":
        return run_prepare(args.data_dir.resolve(), args.results_dir.resolve())
    if args.stage == "test":
        return run_test(args.data_dir.resolve(), args.results_dir.resolve())
    if args.stage == "select":
        return run_select(args.data_dir.resolve(), args.results_dir.resolve())
    return _not_implemented(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
