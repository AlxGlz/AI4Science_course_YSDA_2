"""Проверки загрузчика ``ppg.io`` на синтетических файлах.

Проверяется контракт §3: строковые идентификаторы с ведущими нулями, один
столбец ``signal``, обязательные столбцы разметки, неперекрывающиеся волны и
запрет путей за пределами ``data/``. Каждая проверка порчи данных должна
завершаться явной ошибкой, а не молчаливым чтением.

Данные: временный каталог pytest, настоящая ``data/`` не читается.
Проверка: сам этот файл; запускается ``verify.py``.
Используется в: ``src/ppg/io.py``.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from ppg.io import (
    MANIFEST_COLUMNS,
    SAMPLES_PER_RECORD,
    load_dataset,
    read_annotations,
    read_manifest,
    read_signal,
    sha256_file,
)


def _write_signal(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(values, dtype=np.float64), delimiter=",",
               header="signal", comments="")


def _write_annotations(path: Path,
                       rows: list[tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wave_id", "start_idx", "end_idx",
                         "has_diastolic_peak"])
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, rows: list[dict[str, str]],
                    fieldnames: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames or list(MANIFEST_COLUMNS),
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(root: Path) -> Path:
    """Создать минимальный корректный набор: один эталон и два входа."""
    data = root / "data"
    clean = np.linspace(0.0, 1.0, SAMPLES_PER_RECORD)
    clean_path = data / "references" / "01.csv"
    annotation_path = data / "annotations" / "01.csv"
    _write_signal(clean_path, clean)
    _write_annotations(annotation_path, [
        (0, 100, 420, 1),
        (1, 420, 740, 1),
    ])

    rows: list[dict[str, str]] = []
    for repeat in (1, 2):
        noisy_path = data / "noisy" / f"01-{repeat}.csv"
        _write_signal(noisy_path, clean + 0.01 * repeat)
        rows.append({
            "input_id": f"01-{repeat}",
            "reference_id": "01",
            "noise_repeat": str(repeat),
            "split": "tune",
            "wave_class": "clear",
            "clean_path": "references/01.csv",
            "noisy_path": f"noisy/01-{repeat}.csv",
            "annotation_path": "annotations/01.csv",
            "hr_target_bpm": "60",
            "hr_actual_bpm": "60",
            "period_samples": "320",
            "amplitude": "1",
            "sigma_white": "0.01",
            "drift_amplitude": "0.1",
            "drift_frequency_hz": "0.2",
            "mains_amplitude": "0.01",
            "clean_sha256": _sha(clean_path),
            "noisy_sha256": _sha(noisy_path),
            "annotation_sha256": _sha(annotation_path),
            "dataset_version": "test-v1",
        })
    _write_manifest(data / "manifest.csv", rows)
    return data


def _read_manifest_rows(data: Path) -> tuple[list[str], list[dict[str, str]]]:
    with (data / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_load_dataset_reads_contract_and_keeps_ids(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    dataset = load_dataset(data)

    assert [row.input_id for row in dataset.rows] == ["01-1", "01-2"]
    assert len(dataset.references) == 1
    reference = dataset.references["01"]
    assert reference.split == "tune"
    assert reference.wave_class == "clear"
    assert reference.clean.shape == (SAMPLES_PER_RECORD,)
    assert len(reference.waves) == 2
    assert reference.waves.wave(0) == (100, 420, True)
    assert dataset.by_input("01-2").noise_repeat == 2
    assert len(dataset.rows_for_split("tune")) == 2


def test_read_signal_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    values = np.zeros(10)
    values[3] = np.nan
    _write_signal(path, values)
    with pytest.raises(ValueError, match="NaN"):
        read_signal(path)


def test_read_signal_rejects_wrong_length(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    _write_signal(path, np.zeros(5))
    with pytest.raises(ValueError, match="отсчётов вместо"):
        read_signal(path, expected_samples=SAMPLES_PER_RECORD)


def test_read_annotations_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "annotations.csv"
    _write_annotations(path, [(0, 100, 420, 1), (1, 410, 740, 1)])
    with pytest.raises(ValueError, match="перекрывает"):
        read_annotations(path)


def test_manifest_rejects_missing_column(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    fieldnames, rows = _read_manifest_rows(data)
    fieldnames.remove("noisy_sha256")
    for row in rows:
        row.pop("noisy_sha256", None)
    _write_manifest(data / "manifest.csv", rows, fieldnames=fieldnames)
    with pytest.raises(ValueError, match="обязательных столбцов"):
        read_manifest(data)


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    _, rows = _read_manifest_rows(data)
    rows[0]["noisy_path"] = "../outside.csv"
    _write_manifest(data / "manifest.csv", rows)
    with pytest.raises(ValueError, match="за пределы data"):
        read_manifest(data)


def test_reference_metadata_conflict_is_rejected(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    _, rows = _read_manifest_rows(data)
    rows[1]["split"] = "holdout"
    _write_manifest(data / "manifest.csv", rows)
    with pytest.raises(ValueError, match="противоречит"):
        load_dataset(data)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    payload = b"vasotest-v2\n"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
