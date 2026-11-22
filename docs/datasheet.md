# Паспорт открытых данных `data/` (Datasheet for Datasets)

Оформлен по структуре *Datasheets for Datasets* (Gebru et al., 2021) для закоммиченного открытого набора `data/` версии генератора **2.1.0** (`configs/suite.v2.json`).

## 1. Мотивация (Motivation)

- **Зачем создан набор:** предоставить открытые, полностью воспроизводимые обучающие (`train`) и валидационные (`dev`) эпизоды синтетического финансового аудита, эталонные ответы и записанные траектории базовых политик для разработки и отладки RL-агентов без доступа к закрытым эпизодам жюри (ТЗ разделы 06, 07, 09).
- **Кто создал:** команда участников хакатона Genesis (FinTech, трек 1) с помощью детерминированного генератора `audit_env` (`python -m audit_env dataset data`).

## 2. Состав (Composition)

Точные подсчёты по файлам `data/` версии 2.1.0 (проверяются побайтово в разделе 4 `python -m audit_env verify`):

| Показатель | Значение |
|---|---|
| Всего файлов в `data/` | **188** (187 экспортированных файлов + `data/SHA256SUMS`), общий объём **16 482 258 Б** (~15.7 МБ) |
| Открытых эпизодов | **20** (`train`: **10**, из них 3 чистых; `dev`: **10**, из них 3 чистых) |
| Строк книги (`ledger.csv`) | **7 086** |
| Счетов (`invoices.jsonl`) | **892** |
| Платежей (`payments.jsonl`) | **771** |
| Записей курсов (`fx.csv`) | **3 047** |
| Документов (`documents.jsonl`) | **1 663** |
| Всего эталонных ошибок (`truth.json`) | **40** |
| Строк в `data/trajectories.jsonl` (`expert`) | **314** |
| Траектории 4 базовых политик (`data/trajectories/*.jsonl`) | `oracle.jsonl`: **154**, `reconcile.jsonl`: **314**, `random.jsonl`: **280**, `flag-all.jsonl`: **314** |

Распределение 40 эталонных ошибок по типам в `data/answers/`:
- `fx_rate`: **12**
- `period_cutoff`: **10**
- `payment_allocation`: **8**
- `duplicate_invoice`: **6**
- `vat_on_top`: **2**
- `uncollected_service`: **2**
- Включая оба сценария взаимодействия (`fx_boundary_payment` и `fx_partial_allocation`).

Структура каждого эпизода (`data/episodes/{train,dev}/<episode_id>/`):
- `manifest.json`, `ledger.csv`, `invoices.jsonl`, `payments.jsonl`, `fx.csv`, `policy.json`, `documents.jsonl`;
- отдельно в `data/answers/{train,dev}/<episode_id>/`: `truth.json` и `provenance.json`.
- Сводные файлы: `data/index.json`, `data/benchmark.json`, `data/SHA256SUMS`. Все файлы валидируются по JSON Schema из `schemas/` тестом `tests/test_schema_validation.py`.

## 3. Процесс сбора (Collection Process)

- **Полная синтетика:** данные не собирались из реальных компаний, банков или бухгалтерских баз. Каждый эпизод синтезируется детерминированным конвейером `audit_env/world.py → books.py → scenarios/ → render.py` из `configs/suite.v2.json` на открытом ключе (`keyspace = "open"`).
- **Реальных персональных и финансовых данных нет:** все названия компаний, контрагентов, номера документов, даты, суммы и реквизиты вымышлены.
- **Автоматическая проверка гигиены:** тест `tests/test_episode_io_hygiene.py::HygieneTests::test_no_real_banking_data_in_tracked_files` (выполняется в разделе 1 `python -m audit_env verify`) проверяет отслеживаемые файлы репозитория на отсутствие реальных банковских счетов (IBAN/KZ), БИН и выгрузок банка.

## 4. Предобработка и маркировка (Preprocessing / Labeling)

- Эталон (`truth.json`) не размечается вручную: он вычисляется алгоритмически из причинных переопределений решений бухгалтера (`provenance.json`) с записью изменений «до/после» и SHA-256 чистой книги.
- Независимый аудитор `audit_env/checks.py` подтверждает на каждом эпизоде, что чистая книга (после отката `provenance`) содержит 0 ошибок, а испорченная — ровно эталон `truth.json` (`python -m audit_env verify`, раздел 3; `python -m audit_env audit-truth`).
- Все открытые идентификаторы (`episode_id`, `doc_id`, `row_id`, `invoice_id`, `payment_id`, `fx_id`) — непрозрачные HMAC-строки; строки отсортированы по ним, чтобы порядок не выдавал ошибки.

## 5. Использование (Uses)

- **Рекомендуемое:** отладка агентов и инструментов, офлайн-рескоринг траекторий (`python -m audit_env rescore data/trajectories.jsonl`), обучение базовых ранжировщиков на CPU (`python -m audit_env train`), ручная проверка эталона (`docs/error_catalog.md` + `python tools/catalog_check.py`).
- **Нерекомендуемое:** оценка обобщения по открытым `train`/`dev` (для этого предназначены закрытые шаблоны `configs/holdout-template.v2.json`, `holdout-extended.v2.json`, `curriculum.v2.json` со свежими seed и секретным ключом); перенос выводов на реальный аудит.

## 6. Распространение (Distribution)

- Распространяется в составе репозитория под лицензией **MIT** (`LICENSE`).
- Целостность каждого файла зафиксирована в `data/SHA256SUMS` и проверяется при каждом запуске `python -m audit_env verify` (раздел 4).

## 7. Сопровождение (Maintenance)

- Набор полностью регенерируется одной командой `python -m audit_env dataset data` (с предварительным удалением старой папки при смене версии генератора; см. `PROJECT_STATUS.md`).
- Любое изменение генератора или схем автоматически проверяется CI (`.github/workflows/verify.yml`) на Python 3.11 (Linux, Windows) и в офлайн-контейнере (`make docker-verify`).
