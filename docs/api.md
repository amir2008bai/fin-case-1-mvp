# API для UI

```bash
python -m audit_env serve            # http://127.0.0.1:8765, только localhost, без внешних пакетов
```

Все запросы и ответы — JSON в UTF-8. Для страниц на `http://localhost:*` и `http://127.0.0.1:*` разрешён CORS. Эталон (`truth`) и `provenance` через API не отдаются никогда. Оценка эпизода приходит только в открытом режиме.

| Метод | Путь | Тело | Ответ |
|---|---|---|---|
| GET | `/api/health` | — | `{"ok": true, "version", "sessions"}` |
| POST | `/api/episodes` | `{"seed": 202611, "config": {...}, "mode": "open" \| "closed"}` | `{"session_id", "graded": true, "observation"}` |
| POST | `/api/episodes/import` | `{"folder": "C:/path/to/episode"}` | `{"session_id", "graded": false, "observation"}` или 400 `{"error": "format", "problems": [...]}` |
| GET | `/api/sessions/{id}/files` | — | `{"manifest", "policy", "tables", "documents"}` для просмотра документов |
| POST | `/api/sessions/{id}/step` | `{"action": {"type": "list_records", "table": "ledger", "limit": 50}}` | `{"observation", "reward", "terminated", "truncated", "info"}` |
| POST | `/api/sessions/{id}/run` | `{"policy": "expert" \| "micro" \| "reconcile" \| "random" \| "flag-all"}` | `{"findings", "steps", "seconds", "trajectory", "score"?}` |
| POST | `/api/audit` | `{"folder": "..."}` или `{"manifest", "policy", "tables"}`; можно добавить `"auditor": "rules" \| "micro"` | `{"findings": [...]}` |

`config` устроен так же, как в `audit_env.contracts.Config`:
- `difficulty` — `easy`, `medium` или `hard`;
- `month`, `timezone`;
- `acts`, `customers`, `foreign_percent`;
- `goods_percent` — доля накладных на товары среди всех счетов;
- `vat`;
- `filler` — от 0 до 800 фоновых операций;
- `errors` — включая `vat_on_top` и `uncollected_service`;
- `interactions`, `distractors`;
- `presentation_seed`, `keyspace`.

Неизвестные ключи отклоняются. В режиме `closed` эпизод всегда строится на секретном ключе: эпизод на открытом ключе восстанавливается из открытого кода и seed вместе с ответами, поэтому `"keyspace": "open"` в этом режиме даёт 400.

Документы счетов в `files.documents` различаются полем `document_kind`: `act` — акт выполненных работ (форма Р-1), `waybill` — накладная на отпуск запасов (форма З-2). В поле `title` лежит готовое название для отображения. Если в документе есть НДС, к нему приложен ЭСФ в поле `vat_invoice`.

## Типичный сценарий UI

1. Создать эпизод (`POST /api/episodes`) или загрузить папку с документами (`POST /api/episodes/import`).
2. Показать документы: `GET /api/sessions/{id}/files`.
3. Запустить агента: `POST /api/sessions/{id}/run` с `{"policy": "micro"}`. Каждый вывод содержит `evidence_ids`; по ним UI подсвечивает строки и документы.
4. Чтобы пользователь «добавил или убрал документ», нужно изменить файлы в папке (`invoices.jsonl`, `payments.jsonl`, `ledger.csv`) и снова вызвать `/api/audit` или `/api/episodes/import`. Ошибки формата приходят списком с файлом и строкой.

## Пример (PowerShell)

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/api/episodes -ContentType "application/json" -Body '{"seed": 7, "config": {"keyspace": "open"}, "mode": "open"}'
```
