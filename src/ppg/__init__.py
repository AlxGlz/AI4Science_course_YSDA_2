"""Вычислительные функции проекта: измерение пиков, метрики, фильтрация.

Данные: не читает файлов; массивы передаются вызывающим кодом.
Проверка: ``checks/`` (верификатор ``verify.py``).
Используется в: ``checks/``, далее — ``run_experiment.py`` и
``plot_comparison.py``.
"""

from __future__ import annotations

from .filters import butter_bandpass, moving_average
from .io import (
    CLASS_LABELS,
    SPLIT_LABELS,
    Dataset,
    ManifestRow,
    ReferenceData,
    WaveTable,
    load_dataset,
    read_annotations,
    read_manifest,
    read_signal,
    sha256_file,
)
from .measure import FS_DEFAULT, WaveMeasurement, measure_wave, measure_waves
from .metrics import (
    absolute_errors,
    nested_mean,
    normalized_errors,
    refusal_stats,
)

__all__ = [
    "FS_DEFAULT",
    "WaveMeasurement",
    "measure_wave",
    "measure_waves",
    "absolute_errors",
    "normalized_errors",
    "nested_mean",
    "refusal_stats",
    "butter_bandpass",
    "moving_average",
    "ManifestRow",
    "WaveTable",
    "ReferenceData",
    "Dataset",
    "load_dataset",
    "read_manifest",
    "read_signal",
    "read_annotations",
    "sha256_file",
    "SPLIT_LABELS",
    "CLASS_LABELS",
]
