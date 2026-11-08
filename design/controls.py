"""Контрольные сигналы и ожидаемые ответы для проверок §5 «Верификатор».

Модуль строит детерминированные сигналы аналитически и вычисляет ожидаемые
ответы независимым перебором локальных максимумов (без ``scipy.signal.find_peaks``
и без реализации измерителя из будущего ``src/ppg``).

Данные: нет. Сигналы не читаются с диска, а порождаются формулами §5 методики.
Проверка: ``design/run_design.py`` сверяет ожидаемые ответы с независимым
перебором и с ``scipy.signal.find_peaks`` (кросс-проверка, не источник истины).
Используется в: ``design/run_design.py``; далее — в ``checks/`` верификатора.

Соглашения §2 и §3 методики: частота 320 Гц, индекс начала волны ``s``,
индекс начала следующей волны ``b``, длина волны ``L = b - s``;
для поиска берётся отрезок ``x[s:b+1]``.
"""

from __future__ import annotations

import numpy as np

FS = 320.0                 # Гц, §2 и §3 методики
WAVE_LEN = 320             # отсчётов в волне длительностью 1000 мс
BASELINE = 0.3             # начальный уровень из §5


# --------------------------------------------------------------------------
# Формы: косинусный бугор с финитным носителем, C1-гладкий на краях
# --------------------------------------------------------------------------

def cos_bump(n: int, center: float, half_width: float, amplitude: float) -> np.ndarray:
    """Бугор ``amplitude * (1 + cos(pi * (i - center) / half_width)) / 2``.

    Носитель строго ``center - half_width <= i <= center + half_width``,
    вне носителя точный ноль. В точке ``center`` значение ровно ``amplitude``,
    производные на краях носителя нулевые, поэтому посторонних локальных
    максимумов не возникает.
    """
    i = np.arange(n, dtype=np.float64)
    d = (i - center) / half_width
    out = np.zeros(n, dtype=np.float64)
    mask = np.abs(d) <= 1.0
    out[mask] = amplitude * 0.5 * (1.0 + np.cos(np.pi * d[mask]))
    return out


def two_peak_wave(n: int = WAVE_LEN + 1) -> np.ndarray:
    """CTRL-W1: два раздельных пика, эталон проверки амплитуд и времён §5.

    Волна 320 отсчётов (1000 мс) плюс следующий отсчёт начала новой волны,
    всего 321 отсчёт. Начальный уровень 0,3; максимумы на отсчётах 64 и 192
    со значениями 2,3 и 1,3. Ожидаемые ответы: амплитуды 2 и 1,
    времена 200 и 600 мс.
    """
    x = np.full(n, BASELINE, dtype=np.float64)
    x += cos_bump(n, 64.0, 32.0, 2.0)
    x += cos_bump(n, 192.0, 32.0, 1.0)
    return x


def shoulder_wave(n: int = WAVE_LEN + 1) -> np.ndarray:
    """CTRL-W2: размечен второй пик, но в окне 0,40..0,85 его нет.

    Локальный максимум построения расположен на отсчёте 300 с фазой 0,9375,
    то есть вне окна второго пика. В самом окне сигнал монотонно растёт —
    это «плечо», а не пик. Ожидаемый ответ: отказ измерения второго пика.
    """
    x = np.full(n, BASELINE, dtype=np.float64)
    x += cos_bump(n, 64.0, 32.0, 2.0)
    x += cos_bump(n, 300.0, 60.0, 0.6)
    return x


def single_peak_wave(n: int = WAVE_LEN + 1) -> np.ndarray:
    """CTRL-W3: класс без второго пика (флаг разметки 0).

    Второй пик не размечен, поэтому он не измеряется и ошибка не считается;
    отказом это не является. В окне 0,40..0,85 локальных максимумов нет.
    """
    x = np.full(n, BASELINE, dtype=np.float64)
    x += cos_bump(n, 64.0, 32.0, 2.0)
    x += cos_bump(n, 400.0, 120.0, 0.45)
    return x


def flat_top_wave(n: int = WAVE_LEN + 1) -> np.ndarray:
    """CTRL-W4: плоская вершина из четырёх отсчётов 62..65 на высоте 2,29.

    Правило §2 п.4: для плоской вершины берётся средний отсчёт с округлением
    вниз, то есть floor((62+65)/2) = 63. Округление вверх или к ближайшему
    дало бы 64 — проверка различает эти варианты.
    Соседние отсчёты 61 и 66 равны 2,270031, поэтому вершина действительно
    плоская: четыре равных отсчёта строго выше окружения.
    Ожидаемые ответы: пик на отсчёте 63, амплитуда 1,99, время 196,875 мс.
    """
    x = np.full(n, BASELINE, dtype=np.float64)
    x += cos_bump(n, 63.5, 32.0, 2.0)
    x[62:66] = 2.29
    return x


def prominence_wave(n: int = WAVE_LEN + 1) -> np.ndarray:
    """CTRL-W5: второй пик выбирается по prominence, а не по высоте.

    Профиль задан узлами и C1-интерполяцией без перегибов (PCHIP). Участок
    между первым пиком и вторым окном держится на уровне 1,50, поэтому
    prominence высокого кандидата мала. Локальные максимумы стоят в узлах:

    * отсчёт 190, высота 1,60, prominence 1,60 - max(1,50; 1,58) = 0,02;
    * отсчёт 205, высота 1,62 — самый высокий в окне, prominence 0,12
      (основание 1,50 со стороны первого пика);
    * отсчёт 240, высота 1,10, prominence 1,10 - max(0,80; 0,30) = 0,30.

    Правило §2 п.3 выбирает второй пик по наибольшей prominence → отсчёт 240.
    Выбор по высоте дал бы 205, что и ловится проверкой.
    Ожидаемые ответы: A2 = 1,10 - 0,30 = 0,80, t2 = 750 мс.
    """
    from scipy.interpolate import PchipInterpolator

    knots_x = np.array([0, 32, 64, 100, 130, 160, 190, 200, 205, 215,
                        240, 245, 270, n - 1], dtype=np.float64)
    knots_y = np.array([0.30, 0.30, 2.30, 1.50, 1.50, 1.55, 1.60, 1.58, 1.62,
                        0.80, 1.10, 0.80, 0.30, 0.30], dtype=np.float64)
    f = PchipInterpolator(knots_x, knots_y)
    return f(np.arange(n, dtype=np.float64))


def shifted_first_peak_wave(shift: int, n: int = WAVE_LEN + 1) -> np.ndarray:
    """CTRL-W6a: CTRL-W1 со смещением первого бугра на ``shift`` отсчётов.

    Высота, начальный уровень и длительность волны не меняются.
    При shift = +4 ожидаемые ответы: амплитуда 2, время 212,5 мс;
    относительно CTRL-W1 ошибка времени 12,5 мс, ошибка амплитуды 0.
    """
    x = np.full(n, BASELINE, dtype=np.float64)
    x += cos_bump(n, 64.0 + shift, 32.0, 2.0)
    x += cos_bump(n, 192.0, 32.0, 1.0)
    return x


def shifted_second_peak_wave(shift: int, n: int = WAVE_LEN + 1) -> np.ndarray:
    """CTRL-W6b: CTRL-W1 со смещением второго бугра на ``shift`` отсчётов."""
    x = np.full(n, BASELINE, dtype=np.float64)
    x += cos_bump(n, 64.0, 32.0, 2.0)
    x += cos_bump(n, 192.0 + shift, 32.0, 1.0)
    return x


# --------------------------------------------------------------------------
# Независимый измеритель: только сравнения соседних отсчётов
# --------------------------------------------------------------------------

def local_maxima(x: np.ndarray, s: int, b: int) -> list[int]:
    """Локальные максимумы на отрезке ``x[s:b+1]`` прямым перебором.

    Края ``s`` и ``b`` максимумами не считаются. Плоская вершина из равных
    отсчётов даёт один максимум в среднем отсчёте с округлением вниз.
    """
    peaks: list[int] = []
    i = s + 1
    while i < b:
        if x[i] > x[i - 1]:
            j = i
            while j + 1 <= b and x[j + 1] == x[i]:
                j += 1
            if j + 1 <= b and x[j + 1] < x[i]:
                peaks.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return peaks


def prominence(x: np.ndarray, p: int, s: int, b: int) -> float:
    """Prominence пика ``p``: высота над более высоким из двух оснований.

    Независимая реализация правила §2 п.3: идём влево и вправо до первого
    отсчёта выше вершины (или до края волны), запоминаем минимум на пути;
    основание — максимум из двух минимумов.
    """
    height = float(x[p])
    i = p - 1
    left_min = height
    while i >= s:
        left_min = min(left_min, float(x[i]))
        if x[i] > height:
            break
        i -= 1
    i = p + 1
    right_min = height
    while i <= b:
        right_min = min(right_min, float(x[i]))
        if x[i] > height:
            break
        i += 1
    return height - max(left_min, right_min)


def expected_measurement(x: np.ndarray, has_diastolic_peak: bool,
                         s: int = 0, b: int = WAVE_LEN) -> dict:
    """Ожидаемый результат измерителя §2 по прямому перебору.

    Возвращает словарь с индексами и измеряемыми величинами либо ``None``
    там, где по правилам методики должно быть «не измерять» или отказ.
    """
    L = b - s
    peaks = local_maxima(x, s, b)
    out: dict = {
        "A1": None, "t1": None, "p1": None,
        "A2": None, "t2": None, "p2": None,
        "T": 1000.0 * L / FS,
        "refused_first": False, "refused_second": False,
        "second_not_applicable": not has_diastolic_peak,
    }
    first = [p for p in peaks if 0.10 <= (p - s) / L <= 0.40]
    if not first:
        out["refused_first"] = True
        return out
    # правило §2 п.4: при равной высоте — более ранний максимум
    p1 = max(first, key=lambda p: (float(x[p]), -p))
    out["p1"] = p1
    out["A1"] = float(x[p1] - x[s])
    out["t1"] = 1000.0 * (p1 - s) / FS

    if not has_diastolic_peak:
        return out

    second = [p for p in peaks if p > p1 and 0.40 <= (p - s) / L <= 0.85]
    if not second:
        out["refused_second"] = True
        return out
    scored = [(prominence(x, p, s, b), -p, p) for p in second]
    p2 = max(scored)[2]
    out["p2"] = p2
    out["A2"] = float(x[p2] - x[s])
    out["t2"] = 1000.0 * (p2 - s) / FS
    return out


# --------------------------------------------------------------------------
# Спецификации контрольных сигналов с ожидаемыми ответами §5
# --------------------------------------------------------------------------

def control_specs() -> list[dict]:
    """Список контрольных сигналов с ожидаемыми ответами.

    Поля: ``id``, ``purpose``, ``signal`` (имя построителя), ``x`` (массив),
    ``has_diastolic_peak`` (флаг разметки), ``expected`` (ожидаемые значения
    из независимого перебора), ``note``.
    """
    specs: list[dict] = []

    def add(sid: str, purpose: str, x: np.ndarray, flag: bool,
            note: str = "") -> None:
        specs.append({
            "id": sid,
            "purpose": purpose,
            "x": x,
            "has_diastolic_peak": flag,
            "expected": expected_measurement(x, flag),
            "note": note,
        })

    add("CTRL-W1", "два известных пика: амплитуды 2 и 1, времена 200 и 600 мс",
        two_peak_wave(), True)

    add("CTRL-W2", "плечо: второй пик размечен, но локального максимума в окне нет",
        shoulder_wave(), True, "ожидается отказ измерения второго пика")

    add("CTRL-W3", "класс без второго пика: флаг 0, второй пик не измеряется",
        single_peak_wave(), False, "отказом не является")

    add("CTRL-W4", "плоская вершина 62..65: средний отсчёт с округлением вниз",
        flat_top_wave(), True)

    add("CTRL-W5", "второй пик по prominence, а не по высоте",
        prominence_wave(), True)

    add("CTRL-W6a", "смещение первого пика на +4 отсчёта (12,5 мс)",
        shifted_first_peak_wave(+4), True)
    add("CTRL-W6b", "смещение первого пика на -4 отсчёта (12,5 мс)",
        shifted_first_peak_wave(-4), True)

    return specs
