"""Чтение данных набора ФПГ по договору §3 методики.

Модуль отделяет ввод-вывод от вычислений (§4): здесь только чтение и проверка
структуры ``data/manifest.csv``, CSV сигналов и CSV разметки. Пики, ошибки и
фильтрация живут в ``measure.py``, ``metrics.py`` и ``filters.py``.

Контракт данных (§3 ``METHOD.md``):

* ``manifest.csv`` — 240 строк, идентификаторы хранятся строками с ведущими
  нулями, пути относительны ``data/``;
* сигнал — один столбец ``signal``, 19 200 отсчётов за 60 с при 320 Гц;
* разметка — ``wave_id``, ``start_idx``, ``end_idx``, ``has_diastolic_peak``;
  начало следующей волны равно ``end_idx`` предыдущей.

Данные: ``data/manifest.csv``, ``data/references/*.csv``,
``data/noisy/*.csv``, ``data/annotations/*.csv``.
Проверка: ``checks/test_io.py`` (синтетические файлы, включая намеренно
испорченные) и проверка целостности в ``run_experiment.py prepare``.
Используется в: ``run_experiment.py``, ``plot_comparison.py``,
``src/ppg/__init__.py``.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS_DEFAULT = 320.0
SAMPLES_PER_RECORD = 19_200
SPLITS = ("tune", "holdout")
WAVE_CLASSES = ("clear", "weak", "absent")

SPLIT_LABELS = {
    "tune": "данные для подбора",
    "holdout": "данные для итоговой проверки",
}
CLASS_LABELS = {
    "clear": "небольшой второй максимум",
    "weak": "едва заметный максимум",
    "absent": "плечо без отдельного максимума",
}

MANIFEST_COLUMNS = (
    "input_id", "reference_id", "noise_repeat", "split", "wave_class",
    "clean_path", "noisy_path", "annotation_path", "dataset_version",
    "clean_sha256", "noisy_sha256", "annotation_sha256",
)
ANNOTATION_COLUMNS = ("wave_id", "start_idx", "end_idx", "has_diastolic_peak")
SIGNAL_COLUMN = "signal"


@dataclass(frozen=True)
class ManifestRow:
    """Строка ``manifest.csv`` с проверенными типами.

    Пути хранятся относительными ``data/``; абсолютные пути собирает
    :func:`load_dataset`.
    """

    input_id: str
    reference_id: str
    noise_repeat: int
    split: str
    wave_class: str
    clean_path: str
    noisy_path: str
    annotation_path: str
    dataset_version: str
    clean_sha256: str
    noisy_sha256: str
    annotation_sha256: str


@dataclass(frozen=True)
class WaveTable:
    """Разметка волн одной записи (§3)."""

    start_idx: np.ndarray
    end_idx: np.ndarray
    has_diastolic_peak: np.ndarray

    def __len__(self) -> int:
        return int(self.start_idx.size)

    def wave(self, index: int) -> tuple[int, int, bool]:
        """Вернуть ``(start_idx, end_idx, has_diastolic_peak)`` одной волны."""
        return (
            int(self.start_idx[index]),
            int(self.end_idx[index]),
            bool(self.has_diastolic_peak[index]),
        )


@dataclass(frozen=True)
class ReferenceData:
    """Эталон: чистый сигнал, разметка и пять шумовых входов одной части."""

    reference_id: str
    split: str
    wave_class: str
    clean_path: Path
    annotation_path: Path
    clean: np.ndarray
    waves: WaveTable
    inputs: tuple[ManifestRow, ...]


@dataclass(frozen=True)
class Dataset:
    """Набор данных целиком: 240 входов, сгруппированных по 48 эталонам."""

    data_dir: Path
    rows: tuple[ManifestRow, ...]
    references: dict[str, ReferenceData]

    def by_input(self, input_id: str) -> ManifestRow:
        """Найти вход по ``input_id``; неизвестный идентификатор — ошибка."""
        for row in self.rows:
            if row.input_id == input_id:
                return row
        raise KeyError(f"вход {input_id!r} отсутствует в manifest.csv")

    def rows_for_split(self, split: str) -> tuple[ManifestRow, ...]:
        """Все входы указанной части выборки."""
        if split not in SPLITS:
            raise ValueError(f"неизвестная часть выборки: {split!r}")
        return tuple(row for row in self.rows if row.split == split)


def _require_columns(fieldnames: list[str] | None, required: tuple[str, ...],
                     path: Path) -> None:
    present = set(fieldnames or ())
    missing = [name for name in required if name not in present]
    if missing:
        raise ValueError(f"{path}: нет обязательных столбцов {missing}")


def _parse_int(value: str, name: str, path: Path) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {name}={value!r} не целое число") from exc
    return parsed


def _check_sha256(value: str, name: str, path: Path) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{path}: {name}={value!r} не похоже на SHA-256")


def _resolve_relative(data_dir: Path, relative: str, field: str) -> Path:
    """Собрать абсолютный путь и запретить выход за пределы ``data/``."""
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{field}: путь должен быть относительным: {relative!r}")
    resolved = (data_dir / candidate).resolve()
    if not resolved.is_relative_to(data_dir.resolve()):
        raise ValueError(f"{field}: путь выходит за пределы data/: {relative!r}")
    return resolved


def read_manifest(data_dir: str | Path) -> tuple[ManifestRow, ...]:
    """Прочитать и проверить ``data/manifest.csv`` (§3).

    Идентификаторы остаются строками, поэтому ведущие нули не теряются.
    Проверяются обязательные столбцы, части выборки, классы формы, номера
    повторов и уникальность ``input_id``.
    """
    data_dir = Path(data_dir)
    path = data_dir / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"нет файла манифеста: {path}")

    rows: list[ManifestRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, MANIFEST_COLUMNS, path)
        for line_no, raw in enumerate(reader, start=2):
            input_id = (raw["input_id"] or "").strip()
            reference_id = (raw["reference_id"] or "").strip()
            if not input_id or not reference_id:
                raise ValueError(f"{path}:{line_no}: пустой идентификатор")
            if input_id in seen:
                raise ValueError(f"{path}:{line_no}: повтор input_id={input_id!r}")
            seen.add(input_id)

            split = (raw["split"] or "").strip()
            if split not in SPLITS:
                raise ValueError(f"{path}:{line_no}: неизвестная часть {split!r}")
            wave_class = (raw["wave_class"] or "").strip()
            if wave_class not in WAVE_CLASSES:
                raise ValueError(
                    f"{path}:{line_no}: неизвестный класс {wave_class!r}")

            noise_repeat = _parse_int(raw["noise_repeat"], "noise_repeat", path)
            if noise_repeat < 1:
                raise ValueError(
                    f"{path}:{line_no}: noise_repeat={noise_repeat} < 1")

            for field in ("clean_path", "noisy_path", "annotation_path"):
                _resolve_relative(data_dir, (raw[field] or "").strip(), field)
            if not (raw["dataset_version"] or "").strip():
                raise ValueError(f"{path}:{line_no}: пустой dataset_version")
            for field in ("clean_sha256", "noisy_sha256", "annotation_sha256"):
                _check_sha256((raw[field] or "").strip(), field, path)

            rows.append(ManifestRow(
                input_id=input_id,
                reference_id=reference_id,
                noise_repeat=noise_repeat,
                split=split,
                wave_class=wave_class,
                clean_path=raw["clean_path"].strip(),
                noisy_path=raw["noisy_path"].strip(),
                annotation_path=raw["annotation_path"].strip(),
                dataset_version=raw["dataset_version"].strip(),
                clean_sha256=raw["clean_sha256"].strip(),
                noisy_sha256=raw["noisy_sha256"].strip(),
                annotation_sha256=raw["annotation_sha256"].strip(),
            ))

    if not rows:
        raise ValueError(f"{path}: манифест пуст")
    return tuple(rows)


def read_signal(path: str | Path,
                expected_samples: int | None = None) -> np.ndarray:
    """Прочитать CSV сигнала: один столбец ``signal``, ``float64``.

    ``expected_samples`` задаёт ожидаемую длину записи (в эксперименте —
    19 200). Нечисловые значения и ``NaN``/``inf`` считаются ошибкой данных,
    а не результатом эксперимента (§6).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"нет файла сигнала: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().strip()
    if header != SIGNAL_COLUMN:
        raise ValueError(
            f"{path}: ожидался заголовок {SIGNAL_COLUMN!r}, получено {header!r}")
    try:
        signal = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64,
                            ndmin=1)
    except ValueError as exc:
        raise ValueError(f"{path}: не удалось прочитать сигнал: {exc}") from exc
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size == 0:
        raise ValueError(f"{path}: сигнал пуст")
    if expected_samples is not None and signal.size != expected_samples:
        raise ValueError(
            f"{path}: {signal.size} отсчётов вместо {expected_samples}")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{path}: сигнал содержит NaN или inf")
    return signal


def read_annotations(path: str | Path) -> WaveTable:
    """Прочитать и проверить разметку волн (§3).

    Проверяются столбцы, целые индексы, ``end_idx > start_idx``, отсутствие
    перекрытия волн и флаг ``has_diastolic_peak`` из {0, 1}.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"нет файла разметки: {path}")

    starts: list[int] = []
    ends: list[int] = []
    flags: list[bool] = []
    seen_waves: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, ANNOTATION_COLUMNS, path)
        for line_no, raw in enumerate(reader, start=2):
            wave_id = _parse_int(raw["wave_id"], "wave_id", path)
            if wave_id in seen_waves:
                raise ValueError(f"{path}:{line_no}: повтор wave_id={wave_id}")
            seen_waves.add(wave_id)
            start = _parse_int(raw["start_idx"], "start_idx", path)
            end = _parse_int(raw["end_idx"], "end_idx", path)
            flag = _parse_int(raw["has_diastolic_peak"], "has_diastolic_peak",
                              path)
            if start < 0:
                raise ValueError(f"{path}:{line_no}: start_idx={start} < 0")
            if end <= start:
                raise ValueError(
                    f"{path}:{line_no}: end_idx={end} <= start_idx={start}")
            if starts and start < ends[-1]:
                raise ValueError(
                    f"{path}:{line_no}: волна перекрывает предыдущую "
                    f"(start_idx={start} < {ends[-1]})")
            if flag not in (0, 1):
                raise ValueError(
                    f"{path}:{line_no}: has_diastolic_peak={flag}, ожидалось 0/1")
            starts.append(start)
            ends.append(end)
            flags.append(bool(flag))

    if not starts:
        raise ValueError(f"{path}: разметка пуста")
    return WaveTable(
        start_idx=np.asarray(starts, dtype=np.int64),
        end_idx=np.asarray(ends, dtype=np.int64),
        has_diastolic_peak=np.asarray(flags, dtype=bool),
    )


def sha256_file(path: str | Path) -> str:
    """SHA-256 файла в нижнем регистре — для сверки с ``manifest.csv``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(data_dir: str | Path) -> Dataset:
    """Прочитать манифест, 48 эталонов и 48 разметок в память.

    Шумовые входы читаются по требованию: их 240, и на этапе ``prepare``
    проверяются отдельно. Проверяется единство части, класса и путей у всех
    входов одного эталона, уникальность номеров повторов и границы разметки.
    """
    data_dir = Path(data_dir).resolve()
    rows = read_manifest(data_dir)
    grouped: dict[str, list[ManifestRow]] = {}
    for row in rows:
        grouped.setdefault(row.reference_id, []).append(row)

    references: dict[str, ReferenceData] = {}
    for reference_id, group in grouped.items():
        first = group[0]
        for row in group[1:]:
            for field in ("split", "wave_class", "clean_path",
                          "annotation_path"):
                if getattr(row, field) != getattr(first, field):
                    raise ValueError(
                        f"эталон {reference_id}: вход {row.input_id} "
                        f"противоречит {first.input_id} по {field}")
        repeats = [row.noise_repeat for row in group]
        if len(set(repeats)) != len(repeats):
            raise ValueError(f"эталон {reference_id}: повторяются noise_repeat")

        clean_path = _resolve_relative(data_dir, first.clean_path, "clean_path")
        annotation_path = _resolve_relative(
            data_dir, first.annotation_path, "annotation_path")
        clean = read_signal(clean_path, expected_samples=SAMPLES_PER_RECORD)
        waves = read_annotations(annotation_path)
        if int(waves.end_idx[-1]) > clean.size:
            raise ValueError(
                f"эталон {reference_id}: разметка выходит за запись "
                f"({waves.end_idx[-1]} > {clean.size})")
        references[reference_id] = ReferenceData(
            reference_id=reference_id,
            split=first.split,
            wave_class=first.wave_class,
            clean_path=clean_path,
            annotation_path=annotation_path,
            clean=clean,
            waves=waves,
            inputs=tuple(sorted(group, key=lambda row: row.noise_repeat)),
        )

    return Dataset(data_dir=data_dir, rows=rows, references=references)
