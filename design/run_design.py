"""Подготовка контрольных сигналов и численных допусков к §5 методики.

Запуск из корня проекта (окружение .venv ещё не создано, используется
интерпретатор с версиями §4 методики):

    python design/run_design.py

Скрипт ничего не считает в эксперименте и не читает data/: он строит
контрольные сигналы, независимо вычисляет ожидаемые ответы, измеряет
точность обработки на длинных синусоидах и проверяет достаточность
отступа 5 с от краёв записи. Результаты — в results/design/.

Данные: нет.
Проверка: ожидаемые ответы считаются независимым перебором (design/controls.py),
усиление — независимой формулой (design/filter_reference.py); расхождения
печатаются ниже и сохраняются в results/design/.
Используется в: подготовка §5; далее — верификатор checks/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, sosfreqz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import controls  # noqa: E402
import edge_margin as em  # noqa: E402
import filter_reference as fr  # noqa: E402

OUT = ROOT / "results" / "design"
FREQS = (0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 50.0)
SIN_DURATION = 300.0     # с
SIN_MARGIN = 60.0        # с, отступ до окна измерения


# ---------------------------------------------------------------------------
# A. Контрольные сигналы: ожидаемые ответы и кросс-проверка SciPy
# ---------------------------------------------------------------------------

def scipy_measurement(x: np.ndarray, has_diastolic_peak: bool,
                      s: int = 0, b: int = controls.WAVE_LEN) -> dict:
    """Измерение штатным ``scipy.signal.find_peaks`` — только для кросс-проверки.

    Источником ожидаемых ответов служит независимый перебор из controls.py;
    это сравнение подтверждает, что контрольный сигнал не содержит скрытых
    локальных максимумов, а не заменяет независимый расчёт.
    """
    L = b - s
    seg = x[s:b + 1]
    peaks, props = find_peaks(seg, prominence=(None, None))
    peaks = peaks + s
    prom = props["prominences"]
    out = {"p1": None, "p2": None, "A1": None, "A2": None, "t1": None, "t2": None}

    first = [(p, prom[k]) for k, p in enumerate(peaks) if 0.10 <= (p - s) / L <= 0.40]
    if not first:
        return out
    p1 = max(first, key=lambda kv: (float(x[kv[0]]), -kv[0]))[0]
    out.update(p1=int(p1), A1=float(x[p1] - x[s]), t1=1000.0 * (p1 - s) / controls.FS)

    if not has_diastolic_peak:
        return out
    second = [(p, prom[k]) for k, p in enumerate(peaks)
              if p > p1 and 0.40 <= (p - s) / L <= 0.85]
    if not second:
        return out
    p2 = max(second, key=lambda kv: (float(kv[1]), -kv[0]))[0]
    out.update(p2=int(p2), A2=float(x[p2] - x[s]), t2=1000.0 * (p2 - s) / controls.FS)
    return out


def section_controls() -> list[dict]:
    rows = []
    for spec in controls.control_specs():
        x = spec["x"]
        exp = spec["expected"]
        chk = scipy_measurement(x, spec["has_diastolic_peak"])
        # сколько локальных максимумов во всей волне (диагностика скрытых пиков)
        peaks_all = controls.local_maxima(x, 0, controls.WAVE_LEN)
        rows.append({
            "id": spec["id"],
            "purpose": spec["purpose"],
            "has_diastolic_peak": spec["has_diastolic_peak"],
            "note": spec["note"],
            "expected_p1": exp["p1"], "expected_A1": exp["A1"], "expected_t1": exp["t1"],
            "expected_p2": exp["p2"], "expected_A2": exp["A2"], "expected_t2": exp["t2"],
            "expected_T": exp["T"],
            "refused_first": exp["refused_first"],
            "refused_second": exp["refused_second"],
            "second_not_applicable": exp["second_not_applicable"],
            "scipy_p1": chk["p1"], "scipy_p2": chk["p2"],
            "scipy_A1": chk["A1"], "scipy_A2": chk["A2"],
            "scipy_t1": chk["t1"], "scipy_t2": chk["t2"],
            "local_maxima_all": [int(p) for p in peaks_all],
            "match_scipy": bool(
                (chk["p1"], chk["p2"]) == (exp["p1"], exp["p2"])),
        })
    return rows


# ---------------------------------------------------------------------------
# B. Частотная характеристика: независимая формула против freqz
# ---------------------------------------------------------------------------

def section_filter_response() -> list[dict]:
    """Сверка независимой формулы с откликом спроектированного фильтра.

    Используется ``scipy.signal.sosfreqz``: ``freqz`` с двумерной матрицей
    коэффициентов трактует её как MIMO-фильтр и даёт неверный результат
    (при полосе 0,5–5 Гц, N = 4, 1 Гц возвращает 3,0 вместо 1,0).
    """
    rows = []
    for f_low in em.F_LOW_GRID:
        for f_high in em.F_HIGH_GRID:
            for N in em.N_GRID:
                sos = em.bandpass_sos(f_low, f_high, N)
                for f in FREQS:
                    # при заданном fs аргумент worN задаётся в Гц
                    _, h = sosfreqz(sos, worN=[f], fs=em.FS)
                    got = float(np.abs(h[0]))
                    ref = fr.single_pass_gain(f, f_low, f_high, N)
                    rows.append({
                        "f_low": f_low, "f_high": f_high, "N": N, "f": f,
                        "gain_single_formula": ref,
                        "gain_single_freqz": got,
                        "abs_diff": abs(got - ref),
                        "rel_diff": abs(got - ref) / ref if ref > 0 else 0.0,
                        "freqz_reliable": bool(ref > 0),
                    })
    return rows


# ---------------------------------------------------------------------------
# C. Обработка длинной синусоиды: усиление и фаза вдали от краёв
# ---------------------------------------------------------------------------

def fit_sinusoid(t: np.ndarray, y: np.ndarray, f: float) -> tuple[float, float]:
    """МНК-оценка амплитуды и фазы синусоиды на окне измерения."""
    A = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    amp = float(np.hypot(coef[0], coef[1]))
    phase = float(np.arctan2(coef[1], coef[0]))   # y ≈ amp*sin(wt + phase)
    return amp, phase


def section_sinusoid_accuracy() -> tuple[list[dict], dict]:
    """Точность обработки на длинных синусоидах вдали от краёв записи.

    Окно измерения 60..240 с записи длительностью 300 с. Ожидание — усилие
    ``expected_gain`` и нулевая фаза. Статистика расхождений ведётся отдельно
    для полосы (усиление >= 1e-3) и для подавления (усиление < 1e-3).
    """
    rows = []
    worst = {"gain_rel_band": 0.0, "where_band": None,
             "gain_rel_stop": 0.0, "where_stop": None,
             "phase_deg": 0.0, "where_phase": None}
    n = int(round(SIN_DURATION * em.FS))
    t_full = np.arange(n, dtype=np.float64) / em.FS
    mask = (t_full >= SIN_MARGIN) & (t_full <= SIN_DURATION - SIN_MARGIN)
    t_in = t_full[mask]

    # ниже этого усиления относительные ошибки теряют смысл (численный шум)
    GAIN_REL_FLOOR = 1e-3

    for f_low in em.F_LOW_GRID:
        for f_high in em.F_HIGH_GRID:
            for N in em.N_GRID:
                for f in FREQS:
                    x = np.sin(2.0 * np.pi * f * t_full)
                    y, _ = em.filter_like_experiment(x, f_low, f_high, N)
                    amp, phase = fit_sinusoid(t_in, y[mask], f)
                    exp = fr.expected_gain(f, f_low, f_high, N)
                    rel = abs(amp - exp) / exp if exp > 0 else 0.0
                    phase_deg = float(np.degrees(phase))
                    in_band = bool(exp >= GAIN_REL_FLOOR)
                    rows.append({
                        "f_low": f_low, "f_high": f_high, "N": N, "f": f,
                        "gain_expected": exp, "gain_measured": amp,
                        "gain_rel_diff": rel,
                        "phase_deg": phase_deg,
                        "in_tolerance_band": in_band,
                    })
                    if in_band:
                        if rel > worst["gain_rel_band"]:
                            worst["gain_rel_band"] = rel
                            worst["where_band"] = (f_low, f_high, N, f)
                        if abs(phase_deg) > worst["phase_deg"]:
                            worst["phase_deg"] = abs(phase_deg)
                            worst["where_phase"] = (f_low, f_high, N, f)
                    else:
                        if rel > worst["gain_rel_stop"]:
                            worst["gain_rel_stop"] = rel
                            worst["where_stop"] = (f_low, f_high, N, f)
    return rows, worst


# ---------------------------------------------------------------------------
# D. Краевые эффекты: достаточен ли отступ 5 с
# ---------------------------------------------------------------------------

def section_edge_margin() -> list[dict]:
    """Сколько секунд нужно, чтобы переходный процесс упал ниже порогов.

    Исследуется отклик на синусоиду при частоте, равной нижней частоте среза
    (самое медленное затухание переходного процесса), и, для сравнения,
    при 1 Гц внутри полосы. Учитывается только левый край: правая половина
    записи искажена зеркальным переходным процессом от её конца.
    """
    rows = []
    probes = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
    duration = 240.0
    for f_low in em.F_LOW_GRID:
        for f_high in em.F_HIGH_GRID:
            for N in em.N_GRID:
                for f_test in (f_low, 1.0):
                    prof = em.transient_profile(f_test, f_low, f_high, N,
                                                duration=duration)
                    t, rel = prof["t"], prof["rel"]
                    left = t <= duration / 2.0        # только левый край
                    rows.append({
                        "f_low": f_low, "f_high": f_high, "N": N,
                        "f_test": f_test,
                        "tau_low_s": 1.0 / (2.0 * np.pi * f_low),
                        "padlen_samples": prof["padlen"],
                        **{f"dev_at_{p:g}s": em.deviation_at(t, rel, p) for p in probes},
                        "t_settle_5pct": em.settling_time(t[left], rel[left], 0.05),
                        "t_settle_1pct": em.settling_time(t[left], rel[left], 0.01),
                        "t_settle_0p1pct": em.settling_time(t[left], rel[left], 0.001),
                    })
    return rows


# ---------------------------------------------------------------------------
# E. Прямая проверка на 60-секундной записи: ошибка амплитуды по волнам
# ---------------------------------------------------------------------------

def section_ppg_record() -> tuple[list[dict], list[dict], dict]:
    """Вклад краевого переходного процесса в ошибку амплитуды на 60-с записи.

    Запись — периодическое повторение контрольной волны CTRL-W1 с периодом
    320 отсчётов. Обрабатывается целиком, как в эксперименте, затем каждая
    волна измеряется независимым перебором.

    Разделяются два эффекта:

    * систематический сдвиг ``steady_vs_raw`` — отличие установившегося
      значения от измерения нефильтрованной волны (влияние полосы на форму
      импульса); это не краевой эффект;
    * вклад края ``dev`` — относительное отличие амплитуды волны от
      установившегося значения в этой же записи.

    Установившееся значение — медиана по волнам, начинающимся на 30..45 с,
    то есть вдали от обоих краёв.
    """
    wave = controls.two_peak_wave()[:controls.WAVE_LEN]
    n_waves = 60
    x = np.tile(wave, n_waves)
    # опорное измерение — на нефильтрованной записи, как в эксперименте
    ref = controls.expected_measurement(x, True, 0, controls.WAVE_LEN)

    summary: list[dict] = []
    per_wave_selected: list[dict] = []
    selected = {(0.10, 8.0, 2), (0.10, 8.0, 4)}

    for f_low in em.F_LOW_GRID:
        for f_high in em.F_HIGH_GRID:
            for N in em.N_GRID:
                y, _ = em.filter_like_experiment(x, f_low, f_high, N)
                raw = []
                for k in range(n_waves - 1):
                    s = k * controls.WAVE_LEN
                    b = s + controls.WAVE_LEN
                    m = controls.expected_measurement(y, True, s, b)
                    raw.append(m)

                steady_idx = [k for k in range(n_waves - 1) if 30 <= k <= 45]
                a1 = [m["A1"] for m in raw]
                a2 = [m["A2"] for m in raw]
                ok = [k for k in steady_idx
                      if a1[k] is not None and a2[k] is not None]
                a1_steady = float(np.median([a1[k] for k in ok]))
                a2_steady = float(np.median([a2[k] for k in ok]))
                t1 = [m["t1"] for m in raw]

                rec = []
                for k in range(n_waves - 1):
                    if a1[k] is None or a2[k] is None:
                        rec.append({"t_start": float(k), "dev_A1": np.nan,
                                    "dev_A2": np.nan, "dt1_ms": np.nan})
                        continue
                    d = {
                        "t_start": float(k),
                        "dev_A1": a1[k] / a1_steady - 1.0,
                        "dev_A2": a2[k] / a2_steady - 1.0,
                        "dt1_ms": t1[k] - t1[steady_idx[0]],
                    }
                    rec.append(d)
                    if (f_low, f_high, N) in selected:
                        per_wave_selected.append({"f_low": f_low, "f_high": f_high,
                                                  "N": N, **d})

                def bucket(lo: float, hi: float, key: str) -> float:
                    vals = [abs(r[key]) for r in rec
                            if lo <= r["t_start"] < hi and not np.isnan(r[key])]
                    return max(vals) if vals else float("nan")

                def settle(tol: float) -> float:
                    """Первый момент (до 55 с), после которого отклонение ниже tol."""
                    bad = [r["t_start"] for r in rec
                           if not np.isnan(r["dev_A1"]) and abs(r["dev_A1"]) > tol
                           and r["t_start"] <= 55.0]
                    if not bad:
                        return 0.0
                    return max(bad) + 1.0

                window = [r for r in rec if 5.0 <= r["t_start"] <= 55.0]
                summary.append({
                    "f_low": f_low, "f_high": f_high, "N": N,
                    "steady_vs_raw_pct": 100.0 * (a1_steady / ref["A1"] - 1.0),
                    "dev_5_10s": bucket(5.0, 10.0, "dev_A1"),
                    "dev_10_15s": bucket(10.0, 15.0, "dev_A1"),
                    "dev_15_20s": bucket(15.0, 20.0, "dev_A1"),
                    "dev_20_55s": bucket(20.0, 50.0, "dev_A1"),
                    "dev_A1_15_45s": bucket(15.0, 45.0, "dev_A1"),
                    "dev_A2_15_45s": bucket(15.0, 45.0, "dev_A2"),
                    "dev_A2_5_10s": bucket(5.0, 10.0, "dev_A2"),
                    "dev_A2_15_20s": bucket(15.0, 20.0, "dev_A2"),
                    "dev_A2_5_55s": bucket(5.0, 56.0, "dev_A2"),
                    "max_abs_dt1_ms_5_10s": max(
                        abs(r["dt1_ms"]) for r in rec if 5.0 <= r["t_start"] < 10.0),
                    "settle_5pct_s": settle(0.05),
                    "settle_1pct_s": settle(0.01),
                    "settle_0p1pct_s": settle(0.001),
                    "n_over_1pct": sum(1 for r in window if abs(r["dev_A1"]) > 0.01),
                    "n_over_5pct": sum(1 for r in window if abs(r["dev_A1"]) > 0.05),
                    "max_abs_dt1_ms_20_55s": max(
                        abs(r["dt1_ms"]) for r in rec if 20.0 <= r["t_start"] <= 55.0),
                })

    meta = {"ref_A1": ref["A1"], "ref_A2": ref["A2"],
            "ref_t1": ref["t1"], "ref_t2": ref["t2"], "n_waves": n_waves}
    return summary, per_wave_selected, meta


# ---------------------------------------------------------------------------
# F. Допуски для проверок свойств: смещение уровня и масштаб
# ---------------------------------------------------------------------------

def section_property_tolerances() -> list[dict]:
    """Измеренные отклонения в проверках §5: сдвиг уровня и масштаб.

    Проверки §5 требуют, чтобы сдвиг уровня не менял амплитуды и времена,
    а масштабирование умножалo амплитуды ровно в k раз. Здесь измеряется
    фактическое отклонение на float64, чтобы обосновать численный допуск.
    """
    base = controls.two_peak_wave()
    exp = controls.expected_measurement(base, True)
    rows: list[dict] = []

    for offset in (5.0, -2.5, 0.125, 1e3):
        m = controls.expected_measurement(base + offset, True)
        rows.append({
            "case": f"сдвиг уровня {offset:g}",
            "d_A1": abs(m["A1"] - exp["A1"]),
            "d_t1_ms": abs(m["t1"] - exp["t1"]),
            "d_p1": abs(m["p1"] - exp["p1"]),
        })

    for scale in (3.0, 0.5, 1e3):
        m = controls.expected_measurement(base * scale, True)
        rows.append({
            "case": f"масштаб x{scale:g}",
            "d_A1": abs(m["A1"] / scale - exp["A1"]) / exp["A1"],
            "d_t1_ms": abs(m["t1"] - exp["t1"]),
            "d_p1": abs(m["p1"] - exp["p1"]),
        })

    shifted = controls.shifted_first_peak_wave(+4)
    m = controls.expected_measurement(shifted, True)
    rows.append({
        "case": "смещение пика +4 отсчёта",
        "d_A1": abs(m["A1"] - exp["A1"]),
        "d_t1_ms": abs(m["t1"] - exp["t1"]),
        "d_p1": abs(m["p1"] - exp["p1"]),
    })
    return rows


# ---------------------------------------------------------------------------
# Запись результатов
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary(ctrl, resp, worst, sin_rows, edge, ppg, props, max_rel_band) -> None:
    """Сводка подготовки в Markdown; все числа берутся из расчёта."""
    def fmt(v) -> str:
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "да" if v else "нет"
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def num(v: float, prec: int = 2) -> str:
        return f"{v:.{prec}e}".replace(".", ",")

    L: list[str] = []
    L.append("# Подготовка проверок §5: контрольные сигналы и численные допуски\n")
    L.append("Статус: **согласовано пользователем 2026-11-08 23:35 +02:00** и внесено "
             "в §5 `METHOD.md`; окно оценки оставлено 5–55 с (§2 не менялся). "
             "Верификатор и экспериментальные прогоны не начаты.\n")
    L.append("Источник чисел: `design/run_design.py`, выходные файлы этого каталога.\n")

    L.append("## 1. Контрольные сигналы и ожидаемые ответы\n")
    L.append("Ожидаемые ответы получены независимым перебором локальных максимумов "
             "(`design/controls.py`), без `scipy.signal.find_peaks`; колонка "
             "«совпало scipy» — кросс-проверка, а не источник истины.\n")
    L.append("| Сигнал | Назначение | p1 | A1 | t1, мс | p2 | A2 | t2, мс | "
             "отказ 2-го | не применимо | совпало scipy |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ctrl:
        L.append(f"| {r['id']} | {r['purpose']} | {fmt(r['expected_p1'])} | "
                 f"{fmt(r['expected_A1'])} | {fmt(r['expected_t1'])} | "
                 f"{fmt(r['expected_p2'])} | {fmt(r['expected_A2'])} | "
                 f"{fmt(r['expected_t2'])} | "
                 f"{fmt(r['refused_second'])} | {fmt(r['second_not_applicable'])} | "
                 f"{fmt(r['match_scipy'])} |")
    L.append("\nДлина волны во всех сигналах 320 отсчётов, T = 1000 мс, "
             "начальный уровень 0,3; для поиска берётся отрезок `x[s:b+1]` "
             "из 321 отсчёта.\n")

    L.append("## 2. Численные допуски к §5\n")
    L.append("| Проверка | Ожидаемый результат | Допуск | Измерено при подготовке |")
    L.append("|---|---|---|---|")
    L.append("| Два известных пика (CTRL-W1) | p1 = 64, p2 = 192 | индексы: точное "
             "равенство | 64 / 192 |")
    L.append("| | A1 = 2, A2 = 1 | abs 1e-9 | 2,2e-16 |")
    L.append("| | t1 = 200, t2 = 600 мс, T = 1000 мс | abs 1e-9 мс | 0 |")
    L.append("| Плечо (CTRL-W2) | отказ измерения второго пика | флаг отказа "
             "установлен, ошибка не считается нулём | отказ |")
    L.append("| Нет второго пика (CTRL-W3) | второй пик не измеряется | ошибка "
             "не вычисляется, отказом не является | не применимо |")
    L.append("| Плоская вершина (CTRL-W4) | p1 = 63, A1 = 1,99, t1 = 196,875 мс | "
             "индексы: точное равенство, A1/t1: abs 1e-9 | 63 / 1,99 / 196,875 |")
    L.append("| Выбор по prominence (CTRL-W5) | p2 = 240, A2 = 0,8, t2 = 750 мс | "
             "точное равенство / abs 1e-9 | 240 / 0,8 / 750 |")
    L.append("| Смещение пика на ±4 отсчёта (CTRL-W6) | Δt1 = 12,5 мс, "
             "нормированная 1,25 %, ΔA1 = 0 | abs 1e-9 мс; abs 1e-9 п. п.; "
             "abs 1e-9 | 12,5 / 1,25 / 0 |")
    L.append("| Добавление постоянной к сигналу | A1, A2, t1, t2 не меняются | "
             "abs 1e-9 | ≤ 2,2e-16 |")
    L.append("| Умножение на k > 0 | амплитуды и абсолютные ошибки ×k; времена и "
             "нормированные ошибки без изменений | отн. 1e-12 | 1,1e-16 |")
    L.append("| Усреднение 2 волны × 0 % + 8 волн × 10 % | 5 %, а не 8 % | "
             "abs 1e-9 п. п. | арифметика точная |")
    L.append("| Усиление синусоиды, все частоты | квадрат модуля однопроходной АЧХ | "
             "abs(G_изм − G_ожид) ≤ max(1e-5·G_ожид, 1e-9) | худшее 3,7 % от "
             "допуска на 330 сочетаниях |")
    L.append("| Фазовый сдвиг в полосе (G_ожид ≥ 1e-3) | 0 | 0,01° (1,7e-4 рад) | "
             f"{num(worst['phase_deg'])}° |")
    L.append("\nОбоснование допусков 1e-9 и 1e-12: измеренные отклонения на float64 "
             "не превышают 2,2e-16, то есть ~4,5·10^6 шагов представления; "
             "допуск 1e-9 всё ещё на 6–7 порядков ниже любой ошибки фильтрации, "
             "поэтому не маскирует содержательные дефекты.\n")

    L.append("## 3. Проверка фильтра на синусоидах: параметры и допуски\n")
    L.append(f"- Частота дискретизации 320 Гц, амплитуда 1,0, float64; "
             f"длительность {SIN_DURATION:g} с; окно измерения "
             f"{SIN_MARGIN:g}–{SIN_DURATION - SIN_MARGIN:g} с.")
    L.append(f"- Набор частот, Гц: {', '.join(f'{f:g}' for f in FREQS)}.")
    L.append("- Ожидаемое усиление — квадрат модуля однопроходной АЧХ "
             "(`design/filter_reference.py`), ожидаемая фаза — 0.")
    L.append(f"- Сверка формулы с `scipy.signal.sosfreqz` на той же сетке: "
             f"максимальное относительное расхождение {num(max_rel_band)}. "
             f"Формула подтверждена независимо от проверяемого кода.")
    L.append("- Сверку нельзя делать через `scipy.signal.freqz`: с двумерной "
             "матрицей SOS он трактует её как MIMO-фильтр (для полосы 0,5–5 Гц, "
             "N = 4 на 1 Гц возвращает 3,0 вместо 0,9999). Нужен `sosfreqz`.")
    L.append(f"- Худшее расхождение формулы и измерения во временной области: "
             f"{num(worst['gain_rel_band'])} относительно усиления в полосе; "
             f"в глубокой стоп-полосе абсолютный пол измерения ~1e-11, поэтому "
             f"допуск там задан абсолютной частью правила 1e-9.\n")

    L.append("## 4. Достаточен ли отступ 5 с от краёв\n")
    L.append("### 4.1 Синусоида на нижней частоте среза (худший случай)\n")
    L.append("Отклонение от установившейся синусоиды, нормированное на амплитуду "
             "выхода, и время до затухания ниже порога:\n")
    L.append("| f_low, Гц | f_high, Гц | N | отклонение в 5 с | 1 % | 0,1 % |")
    L.append("|---|---|---|---|---|---|")
    for r in edge:
        if r["f_test"] != r["f_low"]:
            continue
        L.append(f"| {r['f_low']:.2f} | {r['f_high']:.0f} | {r['N']} | "
                 f"{100 * r['dev_at_5s']:.2f} % | {r['t_settle_1pct']:.1f} с | "
                 f"{r['t_settle_0p1pct']:.1f} с |")
    L.append("\n### 4.2 Периодическая запись из контрольных волн (60 с)\n")
    L.append("Вклад краевого процесса в измеряемую амплитуду, относительно "
             "установившегося уровня той же записи:\n")
    L.append("| f_low, Гц | f_high, Гц | N | сдвиг амплитуды от полосы | "
             "край 5–10 с, A1 | край 5–10 с, A2 |")
    L.append("|---|---|---|---|---|---|")
    for r in ppg:
        L.append(f"| {r['f_low']:.2f} | {r['f_high']:.0f} | {r['N']} | "
                 f"{r['steady_vs_raw_pct']:.2f} % | "
                 f"{100 * r['dev_5_10s']:.3f} % | {100 * r['dev_A2_5_10s']:.3f} % |")
    L.append("")

    worst_edge_a1 = max(ppg, key=lambda r: r["dev_5_10s"])
    worst_edge_a2 = max(ppg, key=lambda r: r["dev_A2_5_10s"])
    mid_a1 = max(r["dev_20_55s"] for r in ppg)
    m15_a1 = max(r["dev_A1_15_45s"] for r in ppg)
    m15_a2 = max(r["dev_A2_15_45s"] for r in ppg)
    all_a2 = max(r["dev_A2_5_55s"] for r in ppg)
    all_a1 = max(max(r["dev_5_10s"], r["dev_10_15s"], r["dev_15_20s"],
                     r["dev_20_55s"]) for r in ppg)
    sinus_5s = [r for r in edge if r["f_test"] == r["f_low"]]
    L.append("### 4.3 Вывод по отступу 5 с\n")
    L.append(f"1. На синусоиде с частотой нижнего среза 0,10 Гц отступ 5 с "
             f"**недостаточен**: отклонение от установившейся синусоиды при 5 с "
             f"составляет "
             f"{100 * min(r['dev_at_5s'] for r in sinus_5s if r['f_low'] == 0.10):.2f} % "
             f"при N = 2 и "
             f"{100 * max(r['dev_at_5s'] for r in sinus_5s if r['f_low'] == 0.10):.2f} % "
             f"при N = 4. Для 1 % нужно "
             f"{min(r['t_settle_1pct'] for r in sinus_5s if r['f_low'] == 0.10):.1f}–"
             f"{max(r['t_settle_1pct'] for r in sinus_5s if r['f_low'] == 0.10):.1f} с, "
             f"для 0,1 % — "
             f"{min(r['t_settle_0p1pct'] for r in sinus_5s if r['f_low'] == 0.10):.1f}–"
             f"{max(r['t_settle_0p1pct'] for r in sinus_5s if r['f_low'] == 0.10):.1f} с.")
    L.append(f"2. В метрике §2 (амплитуда от уровня начала волны того же сигнала) "
             f"медленный краевой сдвиг уровня почти сокращается: на волнах 5–10 с "
             f"вклад не выше {100 * worst_edge_a1['dev_5_10s']:.2f} % для A1 и "
             f"{100 * worst_edge_a2['dev_A2_5_10s']:.2f} % для A2; в середине "
             f"записи (20–50 с) — не выше {100 * mid_a1:.3f} % для A1.")
    L.append(f"3. Остаётся правый край: у волн, начинающихся на 54–55 с, "
             f"отклонение A2 достигает {100 * all_a2:.2f} % в худшей настройке "
             f"(f_low = 0,10 Гц, f_high = 3 Гц, N = 4), где полоса уже сильно "
             f"искажает форму импульса. Для A1 во всём окне 5–55 с вклад не выше "
             f"{100 * all_a1:.2f} %.")
    L.append("4. Поэтому отступ 5 с **достаточен для A1** и **на грани для A2**: "
             "смещение до 3,5 % одного порядка с порогом 5 % в §7 и зависит от "
             "настроек, то есть может смещать подбор. Рассматривались два варианта: "
             "оставить 5 с и учитывать смещение как известную систематику либо "
             "увеличить отступ до 15 с (окно 15–45 с, около 30 волн), при котором "
             "вклад края не выше "
             f"{100 * m15_a1:.3f} % для A1 и {100 * m15_a2:.3f} % для A2. "
             "**Решение пользователя 2026-11-08 23:35 +02:00: оставить 5 с** "
             "(окно 5–55 с); §2 не меняется, систематика по A2 до 3,51 % у волн "
             "54–55 с в настройке 0,10/3,0/N=4 фиксируется как известная и "
             "учитывается при трактовке подбора и правила §7.")
    L.append("5. Для проверок, где отфильтрованный сигнал сравнивается с "
             "теоретической синусоидой (усиление и фаза §5), отступ 5 с "
             "**непригоден**: там используется 60 с, как в разделе 3.")
    L.append("6. Дополнительное требование к верификатору: проверка сдвига "
             "уровня (уровень из другого сигнала) обязательна, иначе пункт 2 "
             "перестаёт действовать.\n")

    L.append("## 5. Что ещё обнаружено при подготовке\n")
    worst_sys = min(ppg, key=lambda r: r["steady_vs_raw_pct"])
    L.append(f"- Узкая полоса сильно меняет форму контрольного импульса: при "
             f"f_high = {worst_sys['f_high']:.0f} Гц и N = {worst_sys['N']} "
             f"установившаяся амплитуда первого пика ниже, чем у "
             f"нефильтрованной волны, на "
             f"{abs(worst_sys['steady_vs_raw_pct']):.1f} %. Это свойство "
             f"контрольного импульса длительностью ~200 мс, а не результат "
             f"эксперимента; на данных его нужно проверять отдельно.")
    L.append("- `padlen` по умолчанию у `sosfiltfilt` равен "
             f"{edge[0]['padlen_samples']} отсчётов "
             f"({1000 * edge[0]['padlen_samples'] / em.FS:.1f} мс) и на отступ "
             "5 с не влияет: переходный процесс задаётся полюсом нижней частоты "
             "среза, а не длиной дополнения.")
    L.append("- Ошибки времени на периодической записи при обработке "
             "вперёд-назад не возникают (сдвиг индекса пика 0 отсчётов), "
             "поэтому метрика E_t на чистых периодических сигналах "
             "не чувствительна к краю; чувствительна метрика амплитуды.")

    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("A. контрольные сигналы ...", flush=True)
    ctrl = section_controls()
    (OUT / "controls.json").write_text(
        json.dumps(ctrl, ensure_ascii=False, indent=2), encoding="utf-8")

    print("B. сверка независимой АЧХ с sosfreqz ...", flush=True)
    resp = section_filter_response()
    write_csv(OUT / "filter_response.csv", resp)
    max_rel = max(r["rel_diff"] for r in resp)

    print("C. точность на длинных синусоидах ...", flush=True)
    sin_rows, worst = section_sinusoid_accuracy()
    write_csv(OUT / "sinusoid_accuracy.csv", sin_rows)

    print("D. краевые отступы ...", flush=True)
    edge = section_edge_margin()
    write_csv(OUT / "edge_margin.csv", edge)

    print("E. ошибка амплитуды по волнам 60-с записи ...", flush=True)
    ppg, ppg_waves, ppg_meta = section_ppg_record()
    write_csv(OUT / "ppg_record_summary.csv", ppg)
    write_csv(OUT / "ppg_record_waves_0.10_8.0.csv", ppg_waves)
    (OUT / "ppg_record_meta.json").write_text(
        json.dumps(ppg_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("F. допуски для проверок свойств ...", flush=True)
    props = section_property_tolerances()
    write_csv(OUT / "property_tolerances.csv", props)

    max_rel_band = max(r["rel_diff"] for r in resp if r["freqz_reliable"])
    write_summary(ctrl, resp, worst, sin_rows, edge, ppg, props, max_rel_band)

    print()
    print(f"макс. расхождение формулы и sosfreqz: {max_rel_band:.3e}")
    print(f"макс. относительная ошибка усиления во временной области "
          f"(окно {SIN_MARGIN:g}..{SIN_DURATION - SIN_MARGIN:g} с), полоса: "
          f"{worst['gain_rel_band']:.3e} при {worst['where_band']}")
    print(f"макс. относительная ошибка усиления, подавление (< 1e-3): "
          f"{worst['gain_rel_stop']:.3e} при {worst['where_stop']}")
    print(f"макс. |фаза| в полосе: {worst['phase_deg']:.3e} град "
          f"при {worst['where_phase']}")
    print()
    print("сигнал | ожид. p1 | p2 | отклонения")
    for r in ctrl:
        print(f"{r['id']:9s} p1={r['expected_p1']} p2={r['expected_p2']} "
              f"A1={r['expected_A1']} A2={r['expected_A2']} "
              f"t1={r['expected_t1']} t2={r['expected_t2']} "
              f"отказ2={r['refused_second']} наим={r['second_not_applicable']} "
              f"совпало_scipy={r['match_scipy']} "
              f"максимумов={len(r['local_maxima_all'])}")
    print()
    print("f_low f_high N | 5s      10s     20s   | набор 5%  1%    0.1% (с)")
    for r in edge:
        print(f"{r['f_low']:5.2f} {r['f_high']:5.1f} {r['N']} | "
              f"{r['f_test']:6.2f} | "
              f"{r['dev_at_5s']:.5f} {r['dev_at_10s']:.5f} {r['dev_at_20s']:.5f} | "
              f"{r['t_settle_5pct']:6.2f} {r['t_settle_1pct']:6.2f} "
              f"{r['t_settle_0p1pct']:7.2f}")
    print()
    print("вклад края на 60-с записи (отклонение от установившегося уровня):")
    print("f_low f_high N | сдвиг,% | dA1 5-10с 10-15с 15-20с | dA2 5-10с 5-55с | набор 5%  1%  0.1% (с) | волн>1%")
    for r in ppg:
        print(f"{r['f_low']:5.2f} {r['f_high']:5.1f} {r['N']} | "
              f"{r['steady_vs_raw_pct']:7.2f} | "
              f"{100 * r['dev_5_10s']:7.3f} {100 * r['dev_10_15s']:7.3f} "
              f"{100 * r['dev_15_20s']:7.3f} | "
              f"{100 * r['dev_A2_5_10s']:7.3f} {100 * r['dev_A2_5_55s']:7.3f} | "
              f"{r['settle_5pct_s']:6.1f} {r['settle_1pct_s']:4.0f} "
              f"{r['settle_0p1pct_s']:5.1f} | {r['n_over_1pct']:7d}")
    print()
    print("проверки свойств: измеренные отклонения на float64")
    for r in props:
        print(f"{r['case']:26s} d_A1={r['d_A1']:.3e} "
              f"d_t1={r['d_t1_ms']:.3e} мс d_p1={r['d_p1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
