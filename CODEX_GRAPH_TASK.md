# Задача для Codex: схемы системы в репозитории

Репозиторий: `amir2008bai/fin-case-1-mvp`, ветка `claude/awesome-ptolemy-7ednnx`. Хакатон Genesis, FinTech, трек 1: офлайн RL-среда для ИИ-аудитора.

## Цель

Создать `docs/architecture.md` с набором Mermaid-диаграмм (GitHub рисует их сам, без картинок и внешних сервисов). Читатель — жюри: за 2 минуты понять, как устроена среда, что видит агент, как считается награда и почему её нельзя обмануть. В README после блока «Кратко» добавить одну ссылку: «Схема системы → docs/architecture.md».

## Сначала прочитай (обязательно, не выдумывай связи)

`AGENTS.md`, `PROJECT_STATUS.md`, `README.md`, `docs/design.md`, `docs/protocol.md`, `docs/api.md`, затем код: `audit_env/__main__.py` (команды), `environment.py`, `generator.py`, `world.py`, `scenarios/`, `checks.py`, `scoring.py`, `keys.py`, `runner.py`, `api.py`, `mcp_server.py`, `gym_adapter.py`, `attacks.py`, `leakage.py`, `jury.py`. Каждую стрелку на схеме подтверди импортом или вызовом в коде.

## Диаграммы (по одной на раздел, под каждой 2–4 строки пояснения и ссылки на файлы)

1. **Карта модулей** (`flowchart LR`), группы через `subgraph`:
   - Генерация: `contracts.py` (Config) → `generator.py` → `world.py` + `scenarios/*` (duplicate_invoice, payment_allocation, fx_rate, period_cutoff, real_practice = vat_on_top и uncollected_service, interactions) → `books.py`, `fx.py`, `render.py` → эпизод (файлы) + truth (эталон).
   - Ключи: `keys.py` — открытый keyspace (воспроизводимые обучающие эпизоды) и секретный (закрытые; ключ только в памяти, если не задан AUDIT_ENV_SECRET / AUDIT_ENV_KEY_FILE).
   - Среда: `environment.py` (reset/step, 5 инструментов, лимит 100 шагов) → `scoring.py` (matcher и награда) ← truth.
   - Независимая проверка: `checks.py` пересчитывает эталон по `policy.py` без генератора.
   - Агенты: `agents.py` (expert, micro, reconcile, random, flag-all), `micro.py`, `llm_probe.py`.
   - Интерфейсы: `api.py` (HTTP/JSON + UI `audit_env/ui/`), `mcp_server.py` (для LLM), `gym_adapter.py` (Gymnasium), `runner.py` (подпроцесс/контейнер).
   - Проверки качества: `attacks.py`, `leakage.py`, `economy.py`, `jury.py`, `selfcheck.py`, `diagnostics.py`, `__main__.py verify`.
2. **Жизненный цикл эпизода** (`sequenceDiagram`): Агент → reset(seed, config) → генератор → наблюдение (manifest, задача, политика) → шаги list_records / get_record / get_document / calculator → submit(findings) → scoring → reward. Отметь: в закрытом режиме reward = None для агента, оценка остаётся у оценщика.
3. **Награда и matcher** (`flowchart TD`): вывод агента → проверки лимитов (≤30 выводов, ≤10 доказательств, ≤1000 символов) → сопоставление один-к-одному с эталоном (тип, affected_ids, сумма в допуске, evidence-набор, только виденные ID) → TP/FP/FN → F1 → reward = clip(2·F1 − 1 − 0.002·шаги). Числа сверь с `scoring.py`.
4. **Граница доверия / изоляция** (`flowchart LR` с subgraph «Оценщик» и «Агент»): ключ и truth только у оценщика; недоверенный подпроцесс в закрытом режиме запрещён (`run_subprocess(trusted=False)`), контейнер без сети, своя группа процессов (killpg), переменные с ключом вычищены (`agent_environment()`), API в открытом режиме принимает только открытый keyspace, MCP отказывает секретному без `--closed`. Тесты: `tests/test_adapter_isolation.py`.
5. **Конвейер verify** (`flowchart TD`, 10 разделов): тесты → 22 проверки контракта → независимая проверка эталона → байтовая сверка `data/` → закрытый шаблон (holdout) → расширенный → curriculum → red team (12 атак + перебор seed) → экономия шагов → аудит утечек (AUROC). Названия и порядок возьми из вывода `python -m audit_env verify`.
6. **Кто как подключается** (`flowchart LR`): человек → UI (браузер) → api.py; LLM → MCP → mcp_server.py; RL-алгоритм → gym_adapter.py; скрипт/жюри → CLI (`verify`, `jury`, `attacks`, `llm-probe`, …) и HTTP. Все сходятся в `AuditEnv`.
7. **Типы ошибок** (`mindmap` или `flowchart`): 4 из ТЗ (дубль счёта, распределение платежа, курс/дата курса, отчётный период) + 2 из практики (НДС сверху, продление без оплаты) + 2 взаимодействия (fx_boundary_payment, fx_partial_allocation) + законные «двойники». Ссылка на `docs/error_catalog.md`.

## Правила

- Ничего не меняй в `audit_env/`, `data/`, `models/`, `schemas/`, `configs/`, `tests/`. Только новый `docs/architecture.md` и одна строка-ссылка в README.
- Никаких выдуманных модулей, стрелок и чисел: если связь не нашёл в коде — не рисуй. Числа только из кода/README/вывода команд.
- Диаграммы короткие и читаемые: до ~25 узлов на схему, подписи по-русски, имена файлов в `code`. Цвета через `classDef` (зелёный — оценщик/доверенное, серый — агент, жёлтый — проверки).
- Проверь, что Mermaid валиден: открой файл в превью GitHub или прогони `npx @mermaid-js/mermaid-cli` если доступно; синтаксис без HTML в узлах, кавычки вокруг подписей со скобками.
- После работы: `python -m audit_env verify` (должен остаться зелёным), запись в `PROJECT_STATUS.md` по правилам `AGENTS.md`, коммит в ту же ветку. Этот файл (`CODEX_GRAPH_TASK.md`) после выполнения удали.
