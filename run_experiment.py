"""Этапы эксперимента §6 методики.

Реализованы этапы ``prepare`` (проверка данных и верификатора, план расчётов)
и ``test`` (однократный прогон на одном входе с сохранением измерений,
ошибок, отказов и настроек). Этапы ``select`` и ``main`` пока не реализованы.

Команды (из корня проекта):

    Windows: ``.\\venv\\Scripts\\python.exe run_experiment.py prepare``
             ``.\\venv\\Scripts\\python.exe run_experiment.py test``
    macOS:   ``./.venv/bin/python run_experiment.py prepare``
             ``./.venv/bin/python run_experiment.py test``

Данные: ``data/manifest.csv``, ``data/references/``, ``data/noisy/``,
``data/annotations/``; чтение — через ``ppg.io``, вычисления — через
``ppg.experiment``, ``ppg.filters``, ``ppg.measure``, ``ppg.metrics``.
Проверка: ``verify.py`` (запускается этим скриптом первым шагом).
Используется в: этапы ``select`` и ``main`` после реализации.
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
from ppg.metrics import ABS_KEYS, nested_mean  # noqa: E402

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


def _method_settings(method: dict[str, object]) -> dict[str, object]:
    """Параметры обработки для записи в ``run.json``."""
    keys = ("kind", "window", "f_low_hz", "f_high_hz", "N")
    return {key: method[key] for key in keys if key in method}


def _apply_method(signal, method: dict[str, object]):
    """Применить обработку ко всей записи до выделения волн (§2, §4)."""
    kind = method["kind"]
    if kind == "raw":
        return np.asarray(signal, dtype=np.float64).copy()
    if kind == "moving_average":
        return moving_average(signal, int(method["window"]))
    if kind == "butterworth":
        return butter_bandpass(signal, float(method["f_low_hz"]),
                               float(method["f_high_hz"]), int(method["N"]),
                               fs=FS_HZ)
    raise ValueError(f"неизвестный способ обработки: {kind}")


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
            processed = _apply_method(signal, method)
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
                "settings": _method_settings(method),
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

    test = subparsers.add_parser(
        "test", help="тестовый прогон на одном входе (§6)")
    test.add_argument("--data-dir", type=Path, default=ROOT / "data")
    test.add_argument("--results-dir", type=Path, default=ROOT / "results")

    for name in ("select", "main"):
        subparsers.add_parser(name, help=f"этап {name} (пока не реализован)")

    args = parser.parse_args(argv)
    if args.stage == "prepare":
        return run_prepare(args.data_dir.resolve(), args.results_dir.resolve())
    if args.stage == "test":
        return run_test(args.data_dir.resolve(), args.results_dir.resolve())
    return _not_implemented(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
