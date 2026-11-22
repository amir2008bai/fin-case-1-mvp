# Карточка среды (`Environment Card`): `audit_env` v2.1.0

## 1. Назначение и не-назначение

- **Назначение:** воспроизводимый офлайн RL-симулятор для обучения, диагностики и оценки ИИ-агентов финансового аудита на синтетических бухгалтерских книгах (одна компания, один отчётный месяц, 200–2000 проводок, 20–200 документов, 0–12 скрытых ошибок; `README.md`, `docs/design.md`).
- **Не-назначение:** среда не является бухгалтерской программой, платёжным приложением, чат-ботом или инструментом для реального аудиторского заключения (`README.md`, ТЗ раздел 01 и 12). Высокая награда в среде не доказывает пригодность агента для реального аудита.

## 2. Интерфейс и ограничения

- **Контракт `reset`/`step` (`audit_env/environment.py`, `docs/protocol.md`):**
  - `AuditEnv.reset(seed, config=None) -> (observation, info)` возвращает начальное наблюдение (`task`, `manifest`, `policy`, `tools`, `remaining_steps = 100`) и `info = {}`.
  - `AuditEnv.step(action) -> (observation, reward, terminated, truncated, info)` — строго JSON-совместимый кортеж в стиле Gymnasium.
  - Опциональный адаптер `audit_env/gym_adapter.py` наследует `gymnasium.Env`, если установлен пакет `gymnasium` (в режиме `json_text=True` проходит `gymnasium.utils.env_checker.check_env`; см. `README.md`, `PROJECT_STATUS.md`).
- **Пять инструментов (`docs/protocol.md`, `schemas/action.schema.json`):**
  1. `list_records(table, cursor, limit)` — таблицы `ledger`, `invoices`, `payments`, `fx`; не более 50 строк на страницу, стабильные курсоры.
  2. `get_record(table, id)` — одна запись по первичному ключу (`row_id`, `invoice_id`, `payment_id`, `fx_id`).
  3. `get_document(doc_id)` — полный документ (акт Р-1, накладная З-2, платёжное поручение, ЭСФ, назначение).
  4. `calculator(expression)` — только десятичная арифметика (`Decimal`, 50 знаков), без кода и доступа к файлам.
  5. `submit(findings)` — завершает эпизод (`terminated = True`).
- **Лимиты:** максимум **100 шагов** (100-й шаг — только `submit`) и **60 секунд** на оцениваемый эпизод (`AuditEnv`, `runner.py`). Интерактивные адаптеры (`interactive=True`) поднимают только часы: UI API — 1 час (`audit_env/api.py`), MCP — 30 мин (`audit_env/mcp_server.py`, `--timeout`). Превышение лимита шагов или времени даёт `truncated = True` и оценку пустого ответа.

## 3. Награда и сопоставление (`audit_env/scoring.py`)

- Промежуточная награда всегда `0.0`.
- Ограничения `submit`: до **30** выводов, **1–10** уникальных `evidence_ids`, до **1000** символов `explanation`, `claimed_amount_minor` в диапазоне `[−10¹⁸, 10¹⁸]`. Нарушение структуры завершает эпизод с наградой `−1.0` и кодом `invalid_submission`.
- Сопоставление взаимно однозначное (максимальное паросочетание, augmenting paths): TP требует совпадения `type`, множества `affected_ids`, `currency`, суммы в допуске (`fx_rate`: ±2, остальные: 0) и доказательств (`evidence <= seen`, `evidence <= relevant_ids`, покрытие хотя бы одного допустимого `evidence_set`). Текст `explanation` на оценку не влияет.
- Формула награды: `clip(2·F1 − 1 − 0.002·inspection_steps, −1, 1)` (`P = TP/(TP+FP)`, `R = TP/(TP+FN)`, при пустом эталоне и пустом ответе `F1 = 1`, при пустом эталоне и непустом ответе `F1 = 0`). Отдельно для диагностики доступна экспериментальная `materiality_reward` (`python -m audit_env diagnostics`).

## 4. Режимы `open` / `closed` и изоляция

| Свойство | Открытый режим (`training=True`) | Закрытый режим (`training=False`) |
|---|---|---|
| Ключ по умолчанию | `open` (при `reset(seed)` без конфига и ключа) | строго `secret` (`Config.keyspace = "secret"`) |
| Терминальная награда агенту | возвращается числом | `None` (результат только в `env.outcome`) |
| `truth.json` / `provenance.json` / `seed` | не попадают в наблюдения | не попадают в наблюдения |
| Окружение агента (`runner.py`) | без `AUDIT_ENV_SECRET` и `AUDIT_ENV_KEY_FILE` | без `AUDIT_ENV_SECRET` и `AUDIT_ENV_KEY_FILE`; полная изоляция — контейнер агента (`make docker-agent`) |

## 5. Сплиты и разделение данных (`configs/`, `data/`)

- **Открытые сплиты (`configs/suite.v2.json`, экспортировано в `data/`):** 10 `train` и 10 `dev` эпизодов с фиксированными seed на опубликованном ключе.
- **Закрытые шаблоны без seed (seed и секретный ключ задаёт проверяющий):**
  - `configs/holdout-template.v2.json` — 20 конфигураций (5 чистых, 5 со взаимодействиями, 4 одиночных, 6 смешанных);
  - `configs/holdout-extended.v2.json` — 10 конфигураций (ошибки из практики `vat_on_top`, `uncollected_service` и доли товаров);
  - `configs/curriculum.v2.json` — 18 конфигураций по 6 на уровни `easy` / `medium` / `hard`.
- Пересечение исключено за счёт раздельных конфигураций, независимых seed и секретного ключа (`keyspace="secret"`), который меняет все HMAC-ID, порядок строк и скрытый мир.

## 6. Воспроизводимость

- Одинаковые `seed`, `config`, ключ, версия (`2.1.0`) и действия воспроизводят записи, переходы и награды байт в байт (`tests/test_repro.py`, `tests/test_gaps.py`).
- Все 187 файлов закоммиченного `data/` (без `SHA256SUMS`) сверяются с регенерацией байт в байт в разделе 4 `python -m audit_env verify`.
- Таблицы `benchmark`, `jury`, `attacks`, `economy` и `leakage` выводят секретный ключ из `--seed` (`keys.table_key`), поэтому воспроизводятся точно.

## 7. Известные шорткаты и контроль утечек

Раздел 10 `python -m audit_env verify` (команда `python -m audit_env leakage`) проверяет **360 закрытых эпизодов**:
- **8 признаков оформления** (позиции строк, биты HMAC-ID, секунды меток времени): порог `|AUROC − 0.50| <= 0.06`.
- **7 структурных примет** (`goods_waybill`, `cited_by_receipt`, `receipt_lag_days`, `customer_invoices`, `revenue_posting_time`, `invoice_amount_digit`, `payment_amount_digit`): порог `|AUROC − 0.50| <= 0.08`.
- Ранее обнаруженные приметы (`:00` у пограничных платежей, 4 структурные приметы с AUROC `0.11–0.29` и время суток проводки выручки с ошибкой периода с AUROC `1.0`) устранены в генераторе (`PROJECT_STATUS.md`, `README.md`).

## 8. Результаты базовых политик и диагностики LLM (из `README.md`)

- **Протокол жюри (20 закрытых эпизодов + 20 парных проверок, `python -m audit_env jury`):** `oracle` F1 `1.000` (`+0.980`), `reconcile` F1 `0.201` (`−0.621`), `random` F1 `0.250` (`−0.506`), `flag-all` F1 `0.002` (`−0.997`); `20/20` парных проверок пройдены.
- **Шаблон жюри (`python -m audit_env benchmark --holdout --repeats 5`, 100 эпизодов, 340 ошибок):** `oracle` `+0.962..+1.000`, `expert` и ранжировщик `micro` (поверх правил `checks.py`) `+0.967..+0.972`, `reconcile` `−1.000..−0.381`, `flag-all` `−1.000`.
- **Уровни сложности (`python -m audit_env benchmark --curriculum --repeats 5`, 90 эпизодов):** `reconcile` падает с F1 `0.204` (`easy`) до `0.075` (`hard`), ложных выводов `5.5 → 21.8`/эпизод.
- **Red team (`python -m audit_env attacks`, 108 эпизодов, 360 ошибок):** `honest` `+0.959` на ошибочных, `silence` `+1.000` на чистых; все 12 атак и перебор 200 000 seed (на секретном ключе) не обходят эталоны (`violations: none`).
- **Диагностика настоящих LLM через MCP (`python -m audit_env llm-probe --episodes 5 --same-episodes`, `README.md`, `docs/llm_probe/`):** `deepseek-flash` — 9 из 13 ошибок, 2 FP, F1 `0.82`, награда `+0.54`, 48 шагов; `gemini-3.5-flash-lite` — 0 из 13, F1 `0.20`, награда `−0.60`, 10.6 шагов; `gemini-3.5-flash` (1 чистый) — F1 `1.00`, `+0.98`; `nemotron-3.5-lightning` (1 чистый) — 7 FP, F1 `0.00`, `−1.00`; `expert` — 13 из 13, F1 `1.00`, `+0.98`, 13.2 шагов.
- `micro` — логистический ранжировщик поверх кандидатов и признаков-индикаторов правил `checks.py`, а не самостоятельно обученный аудитор; улучшение модели не заявляется (`README.md`, `docs/micro-model.md`).

## 9. Ограничения, цитирование и лицензия

- **Ограничения (`README.md`):** одна компания и один месяц на эпизод; книга только в KZT; не моделируются полностью отсутствующие в учёте документы и партионный учёт склада; `uncollected_service` строится на подписках, валютные ошибки — на услугах иностранным клиентам; параметры генератора подобраны вручную по практике небольшой сервисной компании, их статистическая репрезентативность не проверялась; аудит утечек проверяет фиксированный список из 12 признаков и не исключает приметы вне него.
- **Лицензия:** MIT (`LICENSE`, `pyproject.toml`).
- **Как цитировать:**
  ```bibtex
  @software{fintech_audit_env_2026,
    title   = {FinTech Case 1: Reproducible RL Environment for Financial Audit (audit\_env)},
    version = {2.1.0},
    year    = {2026},
    license = {MIT},
    note    = {Genesis Hackathon, FinTech Track 1}
  }
  ```
