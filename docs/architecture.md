# Архитектура среды ИИ-аудитора

Генератор моделирует месяц учёта. Агент видит документы и проводки, а оценщик отдельно хранит правильные ответы. Зелёный цвет обозначает доверенные компоненты, серый — сторону агента, жёлтый — проверки.

## 1. Карта модулей

```mermaid
flowchart LR
subgraph G["Генерация"]
C["contracts.py · Config"] --> G0["generator.py"]
G0 --> W["world.py"] --> FX["fx.py"]
G0 --> SC["scenarios/"]
G0 --> B["books.py"]
G0 --> R["render.py"] --> E["Эпизод: таблицы и эталон"]
G0 --> K["keys.py"]
end
subgraph S["Среда"]
ENV["environment.py · AuditEnv"] --> G0
ENV --> SCORE["scoring.py"]
E -->|эталон| SCORE
end
subgraph A["Агенты и интерфейсы"]
P["agents.py · политики"] --> CH["checks.py · эксперт"]
P --> M["micro.py"]
L["llm_probe.py"] --> MCP["mcp_server.py"]
I["api.py / gym_adapter.py / runner.py"]
end
I --> ENV
MCP --> ENV
subgraph Q["Контроль качества"]
CLI["__main__.py"] -->|verify| V["selfcheck.py / attacks.py / economy.py / leakage.py"]
CLI -->|отдельные команды| J["jury.py / diagnostics.py"]
end
V -->|независимая сверка| CH
classDef trusted fill:#e3f4e8,stroke:#477e54,color:#17251a
classDef agent fill:#f1f2f4,stroke:#747a84,color:#202428
classDef check fill:#fff3c7,stroke:#9c7b28,color:#332c18
class ENV,SCORE,E,G0,K trusted
class P,M,L agent
class CH,V,J check
```

[`generator.py`](../audit_env/generator.py) строит мир, применяет сценарии, пересчитывает книги и передаёт результат в [`render.py`](../audit_env/render.py). Запись эпизода и ответов на диск выполняет [`episode_io.py`](../audit_env/episode_io.py).
[`agents.py`](../audit_env/agents.py) предоставляет expert, micro, reconcile, random и flag-all. Эксперт через [`checks.py`](../audit_env/checks.py) независимо выводит ошибки из доступных записей и учётной политики.

## 2. Жизненный цикл эпизода

```mermaid
sequenceDiagram
participant A as Агент или адаптер
participant E as AuditEnv
participant G as Генератор
participant S as Оценщик
A->>E: reset(seed, config)
E->>G: Сформировать эпизод
G-->>E: Таблицы, политика и эталон
E-->>A: manifest, task, policy
loop До submit или исчерпания бюджета
A->>E: list_records / get_record / get_document / calculator
E-->>A: Результат или ошибка, остаток шагов
end
A->>E: submit(findings)
E->>S: Выводы, эталон, виденные ID, шаги
S-->>E: Итоговая оценка
alt Открытый режим
E-->>A: reward
else Закрытый режим
E-->>A: reward = None
end
```

[`environment.py`](../audit_env/environment.py) ограничивает оцениваемый прогон 100 шагами и 60 секундами. Ошибка инструмента тоже расходует шаг.
При завершении закрытого эпизода результат остаётся у оценщика. Формат действий описан в [протоколе](protocol.md).

## 3. Сопоставление и награда

```mermaid
flowchart TD
A["Отчёт агента"] --> B{"Верные episode_id и схема? До 30 выводов, до 10 ID доказательств и 1000 символов пояснения на вывод"}
B -->|нет| X["Награда −1"]
B -->|да| C["Совпадают тип, affected_ids и валюта; сумма в допуске"]
C --> D["Все доказательства видены и относятся к ошибке; есть целиком допустимый набор"]
D --> M["Максимальное сопоставление один к одному"]
M --> F["TP, FP, FN → F1"]
F --> R["reward = clip(2 × F1 − 1 − 0.002 × шаги, −1, 1)"]
classDef trusted fill:#e3f4e8,stroke:#477e54,color:#17251a
classDef agent fill:#f1f2f4,stroke:#747a84,color:#202428
classDef check fill:#fff3c7,stroke:#9c7b28,color:#332c18
class A agent
class B,C,D,M check
class R trusted
```

[`scoring.py`](../audit_env/scoring.py) засчитывает каждую эталонную ошибку один раз. Лишний вывод становится FP, пропуск — FN. Если обе стороны пусты, F1 равен 1.
В стоимость входят шаги осмотра до отправки. Лимиты полей определены в [`contracts.py`](../audit_env/contracts.py).

## 4. Граница доверия

```mermaid
flowchart LR
subgraph T["Оценщик"]
K["keys.py · ключ закрытого набора"] --> G["Генератор"]
G --> H["Скрытый эталон"] --> S["Оценка"]
G --> E["AuditEnv"]
end
subgraph A["Агент"]
D["Контейнер без сети"]
end
E -->|наблюдения| D
D -->|действия и выводы| E
E --> S
classDef trusted fill:#e3f4e8,stroke:#477e54,color:#17251a
classDef agent fill:#f1f2f4,stroke:#747a84,color:#202428
classDef check fill:#fff3c7,stroke:#9c7b28,color:#332c18
class K,G,H,S,E trusted
class D agent
```

[`keys.py`](../audit_env/keys.py) использует открытое пространство ключей для воспроизводимых учебных эпизодов. Секретный ключ по умолчанию живёт в памяти; конфигурация поддерживает `AUDIT_ENV_SECRET` и `AUDIT_ENV_KEY_FILE`.
[`runner.py`](../audit_env/runner.py) запрещает закрытый запуск недоверенного подпроцесса; `agent_environment()` фильтрует окружение, на POSIX `killpg` завершает группу процессов. Границу для чужого агента обеспечивает контейнер.
[`api.py`](../audit_env/api.py) допускает в открытом режиме только открытый keyspace; [MCP](../audit_env/mcp_server.py) требует закрытый режим для секретного набора. [Тесты изоляции](../tests/test_adapter_isolation.py).

## 5. Что проверяет verify

```mermaid
flowchart TD
A["1. Модульные тесты"] --> B["2. Контракт, изоляция и награда · 22 проверки"]
B --> C["3. Независимая сверка эталона на случайных конфигурациях"]
C --> D["4. Побайтовая сверка открытых эпизодов с data/"]
D --> E["5. Закрытый шаблон: базовые агенты, эксперт и парные проверки"]
E --> F["6. Расширенный шаблон: ошибки из практики и товары"]
F --> G["7. Три уровня сложности"]
G --> H["8. Проверка оценщика: 12 стратегий и перебор seed"]
H --> I["9. Экономия шагов"]
I --> J["10. Статистические подсказки: отклонение AUROC от 0,50"]
classDef trusted fill:#e3f4e8,stroke:#477e54,color:#17251a
classDef agent fill:#f1f2f4,stroke:#747a84,color:#202428
classDef check fill:#fff3c7,stroke:#9c7b28,color:#332c18
class A,B,C,D,E,F,G,H,I,J check
```

Порядок задан в [`command_verify`](../audit_env/__main__.py). [`selfcheck.py`](../audit_env/selfcheck.py) проверяет контракт и инварианты, в том числе восстановление чистой книги.
Финальные этапы проверяют [оценщик](../audit_env/attacks.py), [стоимость чтения](../audit_env/economy.py) и [признаки, по которым можно угадать ошибку](../audit_env/leakage.py).

## 6. Кто как подключается

```mermaid
flowchart LR
U["Человек"] --> UI["Браузер · ui/"] --> API["api.py · HTTP/JSON"]
L["LLM"] --> MCP["mcp_server.py · MCP"]
RL["RL-алгоритм"] --> GYM["gym_adapter.py · Gymnasium"]
J["Скрипт или жюри"] --> CLI["CLI · verify, jury, attacks, llm-probe"]
J --> API
CLI -->|llm-probe| MCP
CLI -->|прогоны политик| RUN["runner.py"]
API --> E["AuditEnv"]
MCP --> E
GYM --> E
RUN --> E
classDef trusted fill:#e3f4e8,stroke:#477e54,color:#17251a
classDef agent fill:#f1f2f4,stroke:#747a84,color:#202428
classDef check fill:#fff3c7,stroke:#9c7b28,color:#332c18
class U,L,RL,J agent
class E trusted
```

В [HTTP API](api.md), [MCP](../audit_env/mcp_server.py) и [Gymnasium](../audit_env/gym_adapter.py) действия передаются в одну среду. [CLI](../audit_env/__main__.py) дополнительно запускает проверки и агрегирует результаты.
Для модели доступны пять инструментов. В интерактивной MCP-сессии бюджет времени по умолчанию 30 минут, лимит шагов тот же.

## 7. Какие ошибки расследует агент

```mermaid
flowchart LR
S["Сценарии эпизода"] --> A["4 типа из ТЗ"]
A --> D["Дубль счёта"]
A --> P["Распределение платежа"]
A --> F["Неверный валютный курс"]
A --> T["Неверный отчётный период"]
S --> R["2 типа из практики"]
R --> V["НДС сверх согласованной цены"]
R --> N["Продление услуги без оплаты"]
S --> I["2 взаимодействия"]
I --> X["fx_boundary_payment · валюта на границе периода"]
I --> Y["fx_partial_allocation · частичная валютная оплата"]
S --> Z["Законные двойники: похожие, но корректные операции"]
classDef trusted fill:#e3f4e8,stroke:#477e54,color:#17251a
classDef agent fill:#f1f2f4,stroke:#747a84,color:#202428
classDef check fill:#fff3c7,stroke:#9c7b28,color:#332c18
class S,A,R,I trusted
class Z check
```

[`scenarios/`](../audit_env/scenarios/) содержит модули четырёх обязательных типов, [`real_practice.py`](../audit_env/scenarios/real_practice.py) и [`interactions.py`](../audit_env/scenarios/interactions.py).
[Каталог ошибок](error_catalog.md) показывает проводки и доказательства. Генератор также создаёт чистые книги и похожие законные операции, чтобы проверять ложные обвинения.
