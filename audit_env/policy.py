"""The published accounting policy: every rule an agent needs to recompute the books."""
from .contracts import (ALLOCATION, CUTOFF, DUPLICATE, FX_RATE, MAX_EVIDENCE, SCHEMA_VERSION, TOLERANCE_MINOR,
                        UNCOLLECTED, VAT_ON_TOP)
from .domain import (ACCOUNTS, ADVANCES, BANK, COGS, FX_GAIN, FX_LOSS, INVENTORY, OPENING, RECEIVABLE, REVENUE,
                     VAT)
from .world import STATEMENT_TAIL_DAYS

FINDING_RULES = {
    DUPLICATE: {
        "affected_ids": "invoice_id обоих счетов: исходного и повторного",
        "claimed_amount_minor": "total_minor повторного счёта",
        "currency": "валюта счёта",
        "tolerance_minor": TOLERANCE_MINOR[DUPLICATE],
        "evidence": "doc_id обоих счетов и строки Кт 6010 обоих счетов (или их строки Дт 1210 признания)",
    },
    ALLOCATION: {
        "affected_ids": "payment_id поступления",
        "claimed_amount_minor": "сумма в KZT, которую проводки поступления отнесли на 3510 вместо погашения 1210 "
                                "по правилу распределения",
        "currency": "KZT",
        "tolerance_minor": TOLERANCE_MINOR[ALLOCATION],
        "evidence": "doc_id поступления и счёта, строка Дт 1030 и строка Кт 3510 этого поступления",
    },
    FX_RATE: {
        "affected_ids": "invoice_id, если неверно пересчитан счёт; payment_id, если неверно пересчитано поступление",
        "claimed_amount_minor": "|проведено в KZT − правильно в KZT|: для счёта по сумме Кт 6010, "
                                "для поступления по строке Дт 1030",
        "currency": "KZT",
        "tolerance_minor": TOLERANCE_MINOR[FX_RATE],
        "evidence": "doc_id, fx_id курса правильной даты и затронутая строка (Кт 6010 или Дт 1210 признания "
                    "для счёта, Дт 1030 для поступления)",
    },
    CUTOFF: {
        "affected_ids": "invoice_id, если выручка признана не в том периоде; payment_id, если поступление "
                        "вне периода проведено в периоде",
        "claimed_amount_minor": "для счёта — сумма Кт 6010 по нему в периоде; для поступления — его Дт 1030",
        "currency": "KZT",
        "tolerance_minor": TOLERANCE_MINOR[CUTOFF],
        "evidence": "doc_id и строки Кт 6010 счёта либо строка Дт 1030 поступления",
    },
    VAT_ON_TOP: {
        "affected_ids": "invoice_id счёта, где НДС начислен сверху цены",
        "claimed_amount_minor": "сумма начисленного сверху НДС: Σ(total_minor − quantity × unit_price_minor) по строкам",
        "currency": "валюта счёта",
        "tolerance_minor": TOLERANCE_MINOR[VAT_ON_TOP],
        "evidence": "doc_id счёта и его строки Кт 3130 (можно добавить doc_id оплаты)",
    },
    UNCOLLECTED: {
        "affected_ids": "invoice_id просроченного неоплаченного счёта",
        "claimed_amount_minor": "total_minor этого счёта",
        "currency": "валюта счёта",
        "tolerance_minor": TOLERANCE_MINOR[UNCOLLECTED],
        "evidence": "doc_id просроченного счёта, doc_id нового счёта на ту же услугу и строки Дт 1210 просроченного счёта",
    },
}


def policy_document(world, currencies):
    return {
        "schema_version": SCHEMA_VERSION,
        "period": {
            "start": world.period_start.isoformat(),
            "end": world.period_end.isoformat(),
            "start_inclusive": True,
            "end_exclusive": True,
            "timezone": world.tz_text,
            "rule": "Событие относится к периоду, если его момент времени не раньше start и раньше end. "
                    "Местная дата — дата этого момента в часовом поясе компании (timezone), "
                    "а не дата, записанная в другом смещении.",
        },
        "accounts": ACCOUNTS,
        "revenue_recognition": {
            "date": "service_period_end",
            "rule": f"Выручка по счёту признаётся в день окончания услуги (наибольшая service_period_end "
                    f"по строкам), проводки датируются началом этого дня по времени компании: "
                    f"Дт {RECEIVABLE} Кт {REVENUE} на сумму без НДС и Дт {RECEIVABLE} Кт {VAT} на НДС, "
                    f"по каждой строке. Счёт с окончанием услуги до начала периода входит во входящий "
                    f"остаток (Дт {RECEIVABLE} Кт {OPENING} в момент start) и выручки в периоде не даёт, "
                    f"даже если выписан в периоде. Счёт с окончанием услуги не раньше end выручки "
                    f"в периоде не даёт.",
        },
        "vat": {
            "rate": "0.16",
            "included_in_price": True,
            "rounding": "ROUND_HALF_UP",
            "rule": "НДС включён в цену и считается по каждой строке: total = quantity × unit_price, "
                    "vat = round_half_up(total × 0.16 / 1.16). Строка с vat_rate = null без НДС. "
                    "Валютные счета (экспорт услуг) без НДС.",
        },
        "goods": {
            "documents": "Товары отгружаются по накладной (document_kind = waybill); период строки — дата отгрузки.",
            "rule": f"Выручка по накладной признаётся в дату отгрузки так же, как по акту (Дт {RECEIVABLE} "
                    f"Кт {REVENUE} и Кт {VAT}), себестоимость списывается проводкой Дт {COGS} Кт {INVENTORY} "
                    f"с invoice_id накладной. Остаток товаров на начало периода — Дт {INVENTORY} Кт {OPENING}.",
        },
        "collections": "Счёт, не оплаченный ни на тенге к сроку (due_at), нельзя продлевать: счёт тому же покупателю "
                       "на ту же услугу (service_code) с периодом, начинающимся на следующий день после окончания "
                       "неоплаченного, выписанный после его срока, — нарушение.",
        "fx": {
            "source": "fx.csv",
            "quote": "rate = KZT за 1 единицу source_currency",
            "invoice_rate_date": "service_period_end",
            "payment_rate_date": "местная дата операции поступления",
            "rounding": "ROUND_HALF_UP",
            "formula": "kzt_minor = round_half_up(amount_minor × rate × 10^(exp(KZT) − exp(currency)))",
            "partial_settlement": "Частичная оплата валютного счёта списывает дебиторку пропорционально "
                                  "оплаченной доле (round_half_up), последняя оплата закрывает остаток.",
            "exchange_difference": f"Разница между Дт {BANK} поступления и списанной дебиторкой — "
                                   f"Кт {FX_GAIN} (доход) или Дт {FX_LOSS} (расход). Книга ведётся в KZT.",
        },
        "payments": {
            "timestamp": "Момент банковской операции с тем смещением, в котором его выгрузил банк плательщика.",
            "statement_tail": f"Выписка содержит поступления до {STATEMENT_TAIL_DAYS} дней после end; "
                              f"поступления после end в периоде не проводятся.",
            "positive": f"Поступление от покупателя: Дт {BANK} на всю сумму (в KZT по курсу даты операции).",
            "negative": f"Отрицательная сумма — возврат покупателю ранее полученного аванса: "
                        f"Дт {ADVANCES} Кт {BANK}.",
            "allocation": f"Поступление со ссылками распределяется по счетам в порядке invoice_refs: если выручка "
                          f"по счёту уже признана, погашается его открытый остаток (Кт {RECEIVABLE}, не больше "
                          f"остатка); если ещё не признана — сумма до итога счёта становится авансом по этому "
                          f"счёту (Кт {ADVANCES} с invoice_id) и зачитывается проводкой Дт {ADVANCES} "
                          f"Кт {RECEIVABLE} в момент признания. Остаток сверх всех счетов — аванс без счёта "
                          f"(Кт {ADVANCES}, invoice_id = null).",
            "unreferenced": f"invoice_refs = null: плательщик не указал счёт; вся сумма — Кт {ADVANCES} без счёта. "
                            f"Бухгалтер может позже зачесть её проводкой Дт {ADVANCES} Кт {RECEIVABLE} "
                            f"с указанием счёта и этого payment_id.",
        },
        "opening_balances": f"В момент start: Дт {BANK} Кт {OPENING} — остаток денег; Дт {RECEIVABLE} Кт {OPENING} "
                            f"— открытые счета прошлых периодов; Дт {OPENING} Кт {ADVANCES} — авансы покупателей. "
                            f"Аванс покупателя на начало периода зачитывается (Дт {ADVANCES} Кт {RECEIVABLE}, "
                            f"payment_id = null) при признании первого счёта этого покупателя в KZT.",
        "null": "null в invoice_refs — плательщик не указал счёт; null в entity_id, invoice_id или payment_id "
                "строки книги — у строки нет такой связи.",
        "currencies": sorted(currencies),
        "findings": FINDING_RULES,
        "evidence": f"evidence_ids — doc_id документов, row_id строк книги и fx_id курсов, не больше "
                    f"{MAX_EVIDENCE}. Засчитываются только записи, показанные агенту в этом эпизоде и "
                    f"относящиеся к затронутым документам. Вывод верен, если совпали тип, набор "
                    f"affected_ids, валюта, сумма в пределах допуска и есть хотя бы один полный допустимый "
                    f"набор доказательств.",
    }
