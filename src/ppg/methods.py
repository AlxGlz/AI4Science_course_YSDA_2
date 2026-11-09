"""Применение способов обработки §2 и §4 по описанию из результатов этапа.

Описание метода — словарь с полями ``kind`` (``raw``, ``moving_average``,
``butterworth``) и, в зависимости от способа, ``window``, ``f_low_hz``,
``f_high_hz``, ``N``. Тот же формат сохраняется в ``run.json`` этапов §6,
поэтому графики берут настройки из сохранённых результатов и не подбирают
их заново (§4).

Обработка применяется ко всей записи до выделения волн (§2). Для ``raw``
возвращается копия: вызывающий код не может изменить вход через результат.

Данные: одномерные массивы сигналов ``float64``; файлы не читаются.
Проверка: ``checks/test_methods.py`` (совпадение с ``ppg.filters``,
независимость от входного массива, отказ при неизвестном способе и
некорректных параметрах).
Используется в: ``run_experiment.py`` (этапы ``test``, ``select``, ``main``),
``src/ppg/plotting.py``, ``plot_comparison.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .filters import butter_bandpass, moving_average
from .measure import FS_DEFAULT

METHOD_KEYS = ("kind", "window", "f_low_hz", "f_high_hz", "N")


def method_settings(method: Mapping[str, object]) -> dict[str, object]:
    """Выбрать из описания метода только параметры обработки (§4).

    В ``run.json`` рядом с настройками хранятся имя и подпись; в результаты
    и графики попадают только ``kind``, ``window``, ``f_low_hz``,
    ``f_high_hz`` и ``N``.
    """
    if "kind" not in method:
        raise ValueError(f"в описании метода нет поля 'kind': {method!r}")
    return {key: method[key] for key in METHOD_KEYS if key in method}


def apply_method(signal: np.ndarray, method: Mapping[str, object],
                 fs: float = FS_DEFAULT) -> np.ndarray:
    """Применить обработку ко всей записи по описанию метода (§2, §4).

    ``raw`` — сырой сигнал, ``moving_average`` — центрированное среднее
    с нечётным окном, ``butterworth`` — полосовой Баттерворт вперёд и назад.
    Неизвестный способ и нефинитные значения считаются ошибкой программы
    или данных, а не результатом эксперимента (§6).
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"сигнал должен быть одномерным, получено {x.ndim} осей")
    if x.size == 0:
        raise ValueError("сигнал пуст")
    if not np.all(np.isfinite(x)):
        raise ValueError("сигнал содержит NaN или бесконечность")

    kind = method.get("kind")
    if kind == "raw":
        return x.copy()
    if kind == "moving_average":
        return moving_average(x, int(method["window"]))
    if kind == "butterworth":
        return butter_bandpass(x, float(method["f_low_hz"]),
                               float(method["f_high_hz"]), int(method["N"]),
                               fs=fs)
    raise ValueError(f"неизвестный способ обработки: {kind!r}")
