"""One JSON action per turn for local causal language models."""
import json
from time import perf_counter

from .contracts import MAX_FRAME_BYTES
from .runner import record, strict_loads, wire_frame

SYSTEM_PROMPT = """Вы аудитор синтетической компании. На каждом ходу возвращайте ровно один JSON-объект действия. Расследуйте дубли счетов (duplicate_invoice) и нераспределённые платежи (payment_allocation). Используйте только документы из результатов инструментов.
Инструменты:
{"type":"group_ledger","by":"invoice_id","account":"expense","min_count":2,"limit":12} группирует публичные проводки, а "by":"payment_id","account":"payment_suspense","min_count":1 ищет связанные нераспределённые платежи.
{"type":"search_records","table":"ledger","link":{"field":"invoice_id","id":"<id>"},"limit":12} извлекает источники по связи. Можно искать по payment_id и entity_id или по словам через "query". Таблицы: ledger, invoices, payments, fx.
{"type":"list_records","table":"ledger","limit":20,"cursor":null}, {"type":"get_record","table":"ledger","id":"<id>"}, {"type":"get_document","doc_id":"<id>"} и {"type":"calculator","expression":"1+2"}.
Группировка помогает искать, но доказательства берите из результатов чтения. Суммы в минимальных единицах валюты. Для дубля affected_ids содержит invoice_id, для нераспределённого платежа payment_id. Ссылки должны быть прочитаны до submit.
Завершите действием {"type":"submit","findings":{"episode_id":"<id из манифеста>","findings":[{"type":"duplicate_invoice","affected_ids":["<invoice_id>"],"claimed_amount_minor":123,"currency":"KZT","evidence_ids":["<doc_id>","<row_id>","<row_id>"],"explanation":"Краткое обоснование"}]}}. Если доказательств ошибок нет, отправьте пустой список findings."""


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def message(role, value):
    return {"role": role, "content": value}


def decode_action(raw):
    if not isinstance(raw, str):
        return None
    try:
        if len(raw.encode("utf-8")) > MAX_FRAME_BYTES:
            return None
        value = strict_loads(raw)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


class Conversation:
    def __init__(self, observation):
        self.messages = [
            message("system", SYSTEM_PROMPT),
            message("user", "Начальное наблюдение: " + json_text(observation)),
        ]

    def observe(self, action, observation):
        self.messages += [
            message("assistant", json_text(action)),
            message("user", "Результат инструмента: " + json_text(observation)),
        ]


class ContextLimit(Exception):
    pass


def rollout(env, observation, generate_action, max_actions=100, observer=None):
    """Generate text over public context and return the trusted terminal outcome."""
    if not 1 <= max_actions <= 100:
        raise ValueError("max_actions must be in [1, 100]")
    conversation = Conversation(observation)
    episode_id = observation["manifest"]["episode_id"]

    def emit(step, action, frame, elapsed):
        if observer is not None:
            observer(record(episode_id, step, action, wire_frame(*frame)), elapsed)

    emit(0, {"type": "reset"}, (observation, 0.0, False, False, {}), 0.0)
    for step in range(1, max_actions + 1):
        started = perf_counter()
        try:
            raw = generate_action(conversation.messages)
        except ContextLimit:
            emit(step, None, env.expire("context_limit"), perf_counter() - started)
            return env.outcome
        action = decode_action(raw)
        frame = env.step(action)
        emit(step, action, frame, perf_counter() - started)
        if frame[2] or frame[3]:
            return env.outcome
        conversation.observe(action, frame[0])
    emit(max_actions + 1, None, env.expire("trainer_step_limit"), 0.0)
    return env.outcome
