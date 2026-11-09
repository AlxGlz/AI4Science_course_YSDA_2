"""Независимая сверка результатов этапа ``test`` (§2).

Скрипт не импортирует ``ppg``: сигналы и разметка читаются напрямую,
пики ищутся ``scipy.signal.find_peaks``, формулы амплитуд, времён и
нормированных ошибок записаны заново по тексту §2. Результат сверяется
с ``results/test/measurements.csv`` и ``results/test/errors.csv``.

Команда (из корня проекта):

    .\\venv\\Scripts\\python.exe checks\\evidence\\crosscheck_test_stage.py

Данные: ``data/noisy/08-1.csv``, ``data/references/08.csv``,
``data/annotations/08.csv``, ``results/test/*.csv``.
Проверка: расхождения с сохранёнными результатами печатаются и дают код 1.
Используется в: записи ``DIARY.md`` о тестовом прогоне.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parents[2]
FS = 320.0
TOL = 1e-6


def read_signal(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        next(handle)
        return np.array([float(line) for line in handle], dtype=np.float64)


def read_annotations(path: Path) -> list[tuple[int, int, int, int]]:
    with path.open(encoding="utf-8") as handle:
        return [(int(row["wave_id"]), int(row["start_idx"]), int(row["end_idx"]),
                 int(row["has_diastolic_peak"]))
                for row in csv.DictReader(handle)]


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Центрированное среднее через свёртку (независимо от ``ppg.filters``)."""
    padded = np.pad(signal, (window // 2, window // 2), mode="symmetric")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def butterworth(signal: np.ndarray) -> np.ndarray:
    """Полоса 0,30–8 Гц, N=2 — форма ``b, a`` вместо ``sos``."""
    b, a = butter(2, [0.30, 8.0], btype="bandpass", fs=FS)
    return filtfilt(b, a, signal, padtype="odd", padlen=None)


def measure(x: np.ndarray, s: int, b: int, flag: int) -> dict[str, float | None]:
    segment = x[s:b + 1]
    peaks, props = find_peaks(segment, prominence=(None, None))
    length = b - s
    phases = peaks / length
    first = [i for i in range(peaks.size) if 0.10 <= phases[i] <= 0.40]
    if not first:
        return {"p1": None, "A1": None, "t1": None,
                "p2": None, "A2": None, "t2": None}
    best = max(first, key=lambda i: (segment[peaks[i]], -peaks[i]))
    p1 = int(peaks[best] + s)
    out: dict[str, float | None] = {
        "p1": p1, "A1": float(x[p1] - x[s]),
        "t1": 1000.0 * (p1 - s) / FS, "p2": None, "A2": None, "t2": None}
    second = [i for i in range(peaks.size)
              if 0.40 <= phases[i] <= 0.85 and peaks[i] + s > p1]
    if flag and second:
        chosen = max(second, key=lambda i: (props["prominences"][i],
                                           -peaks[i]))
        p2 = int(peaks[chosen] + s)
        out.update({"p2": p2, "A2": float(x[p2] - x[s]),
                    "t2": 1000.0 * (p2 - s) / FS})
    return out


def main() -> int:
    clean = read_signal(ROOT / "data/references/08.csv")
    noisy = read_signal(ROOT / "data/noisy/08-1.csv")
    annotations = read_annotations(ROOT / "data/annotations/08.csv")
    selected = [item for item in annotations
                if item[1] >= 5 * FS and item[2] <= 55 * FS]
    with (ROOT / "results/test/measurements.csv").open(encoding="utf-8") as handle:
        rows = {(row["method"], int(row["wave_index"])): row
                for row in csv.DictReader(handle)}

    problems: list[str] = []
    checked = 0
    for wave_id, s, b, flag in selected:
        reference = measure(clean, s, b, flag)
        if reference["A1"] is None:
            problems.append(f"волна {wave_id}: эталон не измерен")
            continue
        for method, signal in (
                ("raw", noisy),
                ("moving_average_11", moving_average(noisy, 11)),
                ("butterworth_0.30_8.0_N2", butterworth(noisy))):
            estimate = measure(signal, s, b, flag)
            row = rows[(method, wave_id)]
            expected_norm = {
                "A1": 100.0 * abs(reference["A1"] - estimate["A1"])
                      / reference["A1"],
                "t1": 100.0 * abs(reference["t1"] - estimate["t1"])
                      / (1000.0 * (b - s) / FS),
            }
            if reference["A2"] is not None and estimate["A2"] is not None:
                expected_norm["A2"] = (100.0 * abs(reference["A2"] - estimate["A2"])
                                       / reference["A1"])
                expected_norm["t2"] = (100.0 * abs(reference["t2"] - estimate["t2"])
                                       / (1000.0 * (b - s) / FS))
            for key, value in expected_norm.items():
                saved = float(row[f"norm_{key}"])
                if abs(saved - value) > TOL:
                    problems.append(
                        f"волна {wave_id} {method} {key}: CSV {saved:.9f}, "
                        f"пересчёт {value:.9f}")
            checked += 1
    print(f"Сверено строк: {checked}; волн в окне: {len(selected)}; "
          f"методов: 3")
    if problems:
        print("\n".join(["РАСХОЖДЕНИЯ:"] + problems[:20]))
        return 1
    print("Расхождений нет.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
