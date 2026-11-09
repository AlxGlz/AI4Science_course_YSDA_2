"""Верификатор проекта: запуск ``checks/`` и контроль обязательных частей.

Требования §5 методики:

* запускает pytest для ``checks/`` тем же интерпретатором;
* возвращает код завершения проверок;
* отсутствие обязательного кода или обязательной проверки считается ошибкой
  (код возврата 2), даже если остальные тесты прошли.

Команда (из корня проекта):
    Windows: ``.\\venv\\Scripts\\python.exe verify.py``
    macOS:   ``./.venv/bin/python verify.py``

Данные: не читает ``data/``; проверки работают на порождённых сигналах.
Проверка: сам верификатор; демонстрация пойманной ошибки — в ``DIARY.md``
и ``results/verify/``.
Используется в: ходе эксперимента (§6), перед основным прогоном.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKS = ROOT / "checks"

# Обязательный код: модуль -> требуемые имена (§4 и §5 методики).
REQUIRED_CODE: dict[str, tuple[str, ...]] = {
    "ppg": ("measure_wave", "measure_waves", "butter_bandpass",
            "moving_average", "absolute_errors", "normalized_errors",
            "nested_mean", "refusal_stats"),
    "ppg.measure": ("WaveMeasurement", "measure_wave", "measure_waves"),
    "ppg.metrics": ("absolute_errors", "normalized_errors", "nested_mean",
                    "refusal_stats"),
    "ppg.filters": ("butter_bandpass", "moving_average"),
}

# Обязательные проверки §5: строки таблицы и согласованные контрольные сигналы.
MANDATORY_TESTS: tuple[str, ...] = (
    "checks/test_measure.py::test_control_signals_match_method",
    "checks/test_measure.py::test_known_two_peaks",
    "checks/test_measure.py::test_flat_top_middle_rounded_down",
    "checks/test_measure.py::test_second_peak_chosen_by_prominence",
    "checks/test_measure.py::test_level_shift_invariance",
    "checks/test_measure.py::test_scale_invariance_and_error_scaling",
    "checks/test_measure.py::test_known_time_shift",
    "checks/test_measure.py::test_samples_to_ms_uses_320_hz",
    "checks/test_properties.py::test_level_shift_property",
    "checks/test_metrics.py::test_averaging_equal_weights",
    "checks/test_metrics.py::test_averaging_order_and_duplication",
    "checks/test_metrics.py::test_refusal_stats",
    "checks/test_filters.py::test_sine_gain_and_phase",
    "checks/test_filters.py::test_expected_gain_matches_sosfreqz",
)


def _fail(message: str) -> int:
    print(f"[ОШИБКА] {message}", file=sys.stderr)
    return 2


def check_required_code() -> int:
    """Проверить, что обязательные модули и функции импортируются."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    problems: list[str] = []
    for module_name, names in REQUIRED_CODE.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:                    # noqa: BLE001
            problems.append(f"{module_name}: не импортируется ({exc})")
            continue
        for name in names:
            if not hasattr(module, name):
                problems.append(f"{module_name}: нет обязательного имени {name}")
    if problems:
        for problem in problems:
            print(f"[ОШИБКА] обязательный код: {problem}", file=sys.stderr)
        return 2
    print(f"[OK] обязательный код на месте: {len(REQUIRED_CODE)} модуля")
    return 0


def _pytest(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args,
         "-p", "no:cacheprovider", "--rootdir", str(ROOT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def check_mandatory_tests_registered() -> int:
    """Убедиться, что все обязательные проверки существуют в ``checks/``."""
    if not CHECKS.is_dir():
        return _fail(f"нет каталога проверок {CHECKS}")
    collected = _pytest([str(CHECKS), "--collect-only", "-q"])
    if collected.returncode not in (0, 5):
        sys.stderr.write(collected.stdout)
        sys.stderr.write(collected.stderr)
        return _fail("сбор проверок завершился ошибкой")
    nodeids = {line.strip() for line in collected.stdout.splitlines()
               if "::" in line}
    # параметризованные проверки собираются как ``имя[параметр]``: сверяем
    # базовые идентификаторы, но требуем хотя бы один экземпляр каждого
    base_ids = {nodeid.split("[", 1)[0] for nodeid in nodeids}
    missing = [test for test in MANDATORY_TESTS
               if test.split("[", 1)[0] not in base_ids]
    if missing:
        for test in missing:
            print(f"[ОШИБКА] обязательная проверка не найдена: {test}",
                  file=sys.stderr)
        return 2
    print(f"[OK] обязательных проверок найдено: "
          f"{len(MANDATORY_TESTS)} из {len(MANDATORY_TESTS)}")
    return 0


def main() -> int:
    print("Верификатор: обязательный код → обязательные проверки → pytest")
    for step in (check_required_code, check_mandatory_tests_registered):
        code = step()
        if code != 0:
            print(f"Верификатор остановлен на шаге {step.__name__}.")
            return code

    print("Запуск pytest для checks/ ...")
    result = _pytest([str(CHECKS), "-q"])
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stdout.write(result.stderr)
        print(f"Верификатор: проверки не пройдены, код {result.returncode}.")
    else:
        print("Верификатор: все проверки пройдены, код 0.")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
