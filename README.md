# Проверка фильтрации фотоплетизмограммы

[порождено]

Учебный проект ДЗ2: сравнение сырого зашумлённого сигнала, скользящего среднего
и фильтра Баттерворта по ошибкам амплитуды и времени двух пиков.

## Состояние проекта

- **Этап:** проверки §5 согласованы и внесены в методику; верификатор
  реализован и проходит; загрузчик данных и `run_experiment.py prepare`
  выполнены 2026-11-09 16:47; тестовый прогон `run_experiment.py test`
  выполнен 2026-11-09 17:57, графики `results/test/figures/` построены
  2026-11-09 17:58. Следующий этап — подбор параметров
  `run_experiment.py select`.
- **Готово:** методика, инструкции агенту, CSV данных, исходные разделы отчёта;
  код подготовки проверок (`design/`), сводка с ожидаемыми ответами и
  допусками, исследование краевых эффектов (`results/design/`), дневник,
  подцели и попытки. Контрольные сигналы, численные допуски и параметры
  синусоид внесены в §5 `METHOD.md` (согласовано 2026-11-08 23:35).
  Окружение `.venv/` создано по §4, точные версии зафиксированы. Верификатор:
  `src/ppg/` (измерение пиков, ошибки и усреднение, фильтрация),
  `src/ppg/io.py` (чтение манифеста, сигналов и разметки с проверкой
  контракта §3), `checks/` (контрольные сигналы §5, свойства на Hypothesis,
  АЧХ и фаза, загрузчик на синтетических файлах), `verify.py` запускает
  68 проверок и возвращает код 0. Демонстрация намеренной ошибки 320 → 250
  сохранена в `results/verify/`. `run_experiment.py prepare` проверяет
  SHA-256 и форму 48 эталонов и 240 входов (48/48 и 240/240 без замечаний)
  и сохраняет план расчётов в `results/prepare/`. Тестовый прогон `test`
  на входе `08-1` сохраняет измерения, ошибки и отказы в `results/test/`;
  независимая сверка по сохранённым CSV —
  `checks/evidence/crosscheck_test_stage.py`; относительные ссылки
  в документации — `checks/evidence/check_links.py`. Обработки вынесены
  в `src/ppg/methods.py`, выбор входов этапа и фигуры — в
  `src/ppg/plotting.py`; `plot_comparison.py` строит обзор записи
  и увеличенный фрагмент в `results/test/figures/` (PNG и SVG),
  параметры берёт из `results/test/run.json`.
- **Ещё не выполнено:** `run_experiment.py select`, `run_experiment.py main`,
  подбор параметров и итоговый прогон.
  `REPORT.md` заполнен по подтверждённым результатам; раздел 5 содержит
  таблицы тестового прогона, таблицы подбора и статистики появятся после
  `select` и `main`.
- **Следующий шаг:** `run_experiment.py select` — 30 вариантов Баттерворта,
  четыре окна среднего и сырой сигнал на 24 эталонах для подбора; выбор
  по §7 и фиксация настроек коммитом до `main`.
- **Действующие ограничения:** окно оценки оставлено 5–55 с (§2 не менялся);
  систематика по A2 до 3,51 % у волн 54–55 с (f_low = 0,10 Гц, f_high = 3 Гц,
  N = 4) известна и учитывается при трактовке подбора. Допуски и смысл проверок
  после согласования не подгоняются под результат; изменение критерия требует
  отдельного согласования. Исходные данные сохраняются неизменными.
- **Окружение:** `.venv/` на Python 3.13.14; NumPy 2.5.2, SciPy 1.18.1,
  Matplotlib 3.11.2, statsmodels 0.14.5, pytest 9.1.1, Hypothesis 6.142.4 —
  совпадают с §4. `pip check` чист, точные версии транзитивных зависимостей —
  в `requirements-lock.txt`, версия Python — в `.python-version`.

## Где что лежит

| Путь | Содержание |
|---|---|
| [METHOD.md](METHOD.md) | Методика, измерение пиков, метрики, окружение, команды, этапы и правила сравнения. |
| [AGENTS.md](AGENTS.md) | Инструкции агенту: порядок работы, ограничения и документирование. |
| [REPORT.md](REPORT.md) | Отчёт для преподавателя; заполняется по фактическим результатам. |
| [DIARY.md](DIARY.md) | Дневник: что сделано, какими командами, с каким результатом и какие ошибки пойманы. |
| [SUBGOALS.md](SUBGOALS.md) | Подцели с критериями завершения и командами проверки. |
| [ATTEMPTS.md](ATTEMPTS.md) | Попытки по подцелям, включая неудачные и их причины. |
| [design/](design/) | Код подготовки проверок §5: контрольные сигналы, независимая АЧХ, исследование краевых эффектов. |
| [src/ppg/](src/ppg/) | Вычислительные функции: измерение пиков (§2), ошибки и усреднение, фильтрация (§4). |
| [src/ppg/io.py](src/ppg/io.py) | Чтение `manifest.csv`, сигналов и разметки с проверкой контракта §3; ввод-вывод отделён от вычислений. |
| [src/ppg/experiment.py](src/ppg/experiment.py) | Отбор волн 5–55 с, сравнение обработки с эталоном, ряды ошибок, подсчёт отказов (§2). |
| [src/ppg/methods.py](src/ppg/methods.py) | Применение обработок по сохранённым настройкам; общий код этапов и графиков. |
| [src/ppg/plotting.py](src/ppg/plotting.py) | Выбор входов этапа по §4, общие шкалы амплитуды, отметки начал волн и найденных пиков, сохранение PNG и SVG. |
| [checks/](checks/) | Верификатор: контрольные сигналы §5, проверки свойств на Hypothesis, АЧХ и фаза, чувствительность к ошибкам. |
| [checks/evidence/](checks/evidence/) | Внесение и откат намеренной ошибки 320 → 250; независимая сверка результатов `test` без импорта `ppg`; проверка относительных ссылок в документации. |
| [verify.py](verify.py) | Запуск `checks/` тем же интерпретатором; отсутствие обязательного кода или проверки — ошибка (код 2). |
| [run_experiment.py](run_experiment.py) | Этапы §6; реализованы `prepare` (проверка данных и верификатора, план) и `test` (три обработки на одном входе), `select` и `main` — заготовки. |
| [plot_comparison.py](plot_comparison.py) | Графики по `run.json` этапа: обзор записи и увеличенный фрагмент; параметры обработок не подбирает. |
| [requirements.txt](requirements.txt) | Прямые зависимости с точными версиями `==` по §4. |
| [requirements-lock.txt](requirements-lock.txt) | Полный `pip freeze` окружения, включая транзитивные зависимости. |
| [.python-version](.python-version) | Версия Python окружения (3.13.14). |
| [results/verify/](results/verify/) | Вывод верификатора с намеренной ошибкой и после исправления, unified diff ошибки. |
| [results/prepare/](results/prepare/) | План расчётов, таблицы проверки 48 эталонов и 240 входов, журнал и вывод верификатора. Точка входа — [summary.md](results/prepare/summary.md). |
| [results/test/](results/test/) | Тестовый прогон на входе `08-1`: измерения по волнам, ошибки, отказы, параметры и журнал. Точка входа — [summary.md](results/test/summary.md). |
| [results/test/figures/](results/test/figures/) | Обзор записи и увеличенный фрагмент для входа `08-1` в PNG и SVG. Точка входа — [index.md](results/test/figures/index.md). |
| [results/design/](results/design/) | Результаты подготовки: сводка, таблицы, журнал прогона. Точка входа — [summary.md](results/design/summary.md). |
| [data/](data/) | Только CSV сигналов, разметки и manifest; исходные файлы сохраняются неизменными. |
| [data/references/](data/references/) | 48 чистых эталонов по 60 секунд при 320 Гц. |
| [data/noisy/](data/noisy/) | 240 зашумлённых входов: пять реализаций на эталон. |
| [data/annotations/](data/annotations/) | 48 файлов с границами волн и наличием второго пика. |
| [data/manifest.csv](data/manifest.csv) | Связи входов, эталонов и разметки, разбиение на две части и параметры сигналов. Пути относительны `data/`. |
| [.gitignore](.gitignore) | Исключение `.venv/`, кэшей байткода и служебных файлов из Git. |

В проекте только синтетические данные. Генератор и приватные записи пациентов
в него не перенесены. Для подбора и итоговой проверки выделено по 24 эталона;
все пять реализаций одного эталона относятся к одной части.

### Подготовка проверок §5

| Файл | Содержание |
|---|---|
| [design/controls.py](design/controls.py) | Контрольные сигналы CTRL-W1…CTRL-W6 и ожидаемые ответы независимым перебором. |
| [design/filter_reference.py](design/filter_reference.py) | Ожидаемая АЧХ из параметров методики, независимо от проверяемого кода. |
| [design/edge_margin.py](design/edge_margin.py) | Профиль переходного процесса у краёв записи. |
| [design/run_design.py](design/run_design.py) | Сборка всех результатов подготовки в `results/design/`. |
| [results/design/summary.md](results/design/summary.md) | Сводка: сигналы, ожидаемые ответы, допуски, ответ про отступ 5 с. |
| [results/design/edge_margin.csv](results/design/edge_margin.csv) | Время затухания переходного процесса по настройкам фильтра. |
| [results/design/ppg_record_summary.csv](results/design/ppg_record_summary.csv) | Вклад края и полосы в амплитуду пиков на 60-секундной записи. |
| [results/design/sinusoid_accuracy.csv](results/design/sinusoid_accuracy.csv) | Точность усиления и фазы на длинных синусоидах. |
| [results/design/filter_response.csv](results/design/filter_response.csv) | Сверка независимой АЧХ с `sosfreqz`. |
| [results/design/run_log.txt](results/design/run_log.txt) | Полный вывод прогона подготовки. |

## Что появится при реализации

- `run_experiment.py select`, `main` — подбор параметров и основной прогон;
  сейчас реализованы `prepare` и `test`.
- `results/select/`, `results/main/` — таблицы, параметры и графики этапов.

Это планируемые части, а не уже выполненная работа. После их создания агент
обновляет эту карту ссылками на реальные файлы.

## Верификатор

| Файл | Что проверяет |
|---|---|
| [checks/test_measure.py](checks/test_measure.py) | CTRL-W1…W6 по таблице §5: пики, плоская вершина, prominence, отказы, сдвиг уровня, масштаб, известное смещение ±4 отсчёта. |
| [checks/test_properties.py](checks/test_properties.py) | Свойства на порождённых входах (Hypothesis, `derandomize=True`): уровень, масштаб, сдвиг времени, порядок строк. |
| [checks/test_metrics.py](checks/test_metrics.py) | Усреднение 2×0 % + 8×10 % = 5 %, инвариантность к порядку и дублированию, доля отказов. |
| [checks/test_filters.py](checks/test_filters.py) | Усиление и фаза на синусоидах 300 с, сверка формулы АЧХ с `sosfreqz`, скользящее среднее. |
| [checks/test_io.py](checks/test_io.py) | Загрузчик на синтетических файлах: строковые ID, отказ при `NaN`, неверной длине, перекрытии волн, отсутствующем столбце и выходе за `data/`. |
| [checks/test_experiment.py](checks/test_experiment.py) | Отбор волн 5–55 с, ошибки на сдвинутом и масштабированном сигнале, отказы вместо нулевой ошибки, неприменимый второй пик. |
| [checks/test_methods.py](checks/test_methods.py) | Применение обработок: совпадение с `ppg.filters`, независимость результата от входа, отказ при неизвестном способе и нефинитных значениях. |
| [checks/test_plotting.py](checks/test_plotting.py) | Выбор входов по этапу и классу, запрет итоговой части для подбора, общие шкалы, отметки волн и пиков, сохранение PNG и SVG. |
| [checks/reference.py](checks/reference.py) | Независимые ожидаемые ответы: `design/controls.py`, `design/filter_reference.py` и таблица §5. |
| [results/verify/](results/verify/) | Доказательство: вывод с ошибкой 320 → 250 и после исправления. |

## Воспроизведение

Версии пакетов и команды для Windows и macOS заданы в
[§4 методики](METHOD.md#4-код-и-окружение), порядок этапов — в
[§6](METHOD.md#6-этапы-эксперимента).

Подготовка проверок §5 (не требует `.venv/` и не читает `data/`):

```powershell
# Windows
python design\run_design.py
```

```bash
# macOS
python3 design/run_design.py
```

Запись пересчитывает `results/design/` заново; длительность около 9 с.

Верификатор (требует `.venv/`; `data/` не читает):

```powershell
# Windows
.\.venv\Scripts\python.exe verify.py
```

```bash
# macOS
./.venv/bin/python verify.py
```

Код возврата 0 — все проверки пройдены; 1 — проверка не пройдена;
2 — отсутствует обязательный код или обязательная проверка.
Проверки свойств зафиксированы с `derandomize=True`, поэтому прогон детерминирован.

Подготовка эксперимента `prepare` (§6; читает `data/`, проверяет SHA-256 и форму
всех 288 файлов, запускает верификатор и сохраняет план в `results/prepare/`):

```powershell
# Windows
.\.venv\Scripts\python.exe run_experiment.py prepare
```

```bash
# macOS
./.venv/bin/python run_experiment.py prepare
```

Код возврата 0 — данные и верификатор в порядке, план сохранён; 1 — ошибка
данных; иной код — остановка на верификаторе. Длительность около 9 с.

Тестовый прогон `test` (§6; один вход `08-1`, три обработки, окно оценки 5–55 с):

```powershell
# Windows
.\.venv\Scripts\python.exe run_experiment.py test
```

```bash
# macOS
./.venv/bin/python run_experiment.py test
```

Код возврата 0 — расчёт завершён, результаты в `results/test/`; 1 — данные или
расчёт не позволили продолжить; иной код — остановка на верификаторе.
Длительность около 8 с. Независимая сверка сохранённых CSV:

```powershell
.\.venv\Scripts\python.exe checks\evidence\crosscheck_test_stage.py
```

Графики тестового прогона (§4; читают `results/test/run.json`, параметры
обработок не подбирают; обзор записи и фрагмент в `results/test/figures/`):

```powershell
# Windows
.\.venv\Scripts\python.exe plot_comparison.py --results results/test
```

```bash
# macOS
./.venv/bin/python plot_comparison.py --results results/test
```

Код возврата 0 — по каждому входу сохранены PNG и SVG; 1 — нет `run.json`,
вход не рассчитан на этапе или интервал выходит за запись. Длительность
около 2 с. Вход и интервал задаются `--input-id`, `--start`, `--duration`
(время в секундах), выбор обработок — `--method`.

Демонстрация намеренной ошибки (из корня проекта, Windows):

```powershell
.\.venv\Scripts\python.exe checks\evidence\inject_320_to_250.py --apply
.\.venv\Scripts\python.exe verify.py        # ожидается код 1
.\.venv\Scripts\python.exe checks\evidence\inject_320_to_250.py --revert
.\.venv\Scripts\python.exe verify.py        # ожидается код 0
```

Сохранённый вывод: [verify_bug_320_to_250.log](results/verify/verify_bug_320_to_250.log),
[verify_after_fix.log](results/verify/verify_after_fix.log).
Проверка относительных ссылок в документации (код 0 — битых ссылок нет):

```powershell
.\.venv\Scripts\python.exe checks\evidence\check_links.py
```

Остальные команды станут доступны после реализации `select` и `main`.
