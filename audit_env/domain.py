"""Chart of accounts and currencies of the synthetic company.

Account codes follow the Kazakhstan standard chart of accounts; only the accounts
the episode files need are used. Ledger amounts are in the functional currency.
"""
from decimal import Decimal

BASE_CURRENCY = "KZT"
CURRENCY = BASE_CURRENCY
MINOR_UNITS = {"KZT": 2, "USD": 2, "RUB": 2}
FOREIGN_CURRENCIES = ("USD", "RUB")
VAT_RATE = Decimal("0.16")

BANK = "1030"
RECEIVABLE = "1210"
VAT = "3130"
PAYABLE = "3310"
ADVANCES = "3510"
OPENING = "5610"
REVENUE = "6010"
FX_GAIN = "6250"
EXPENSE = "7210"
FX_LOSS = "7430"

ACCOUNTS = {
    BANK: "Денежные средства на текущих банковских счетах",
    RECEIVABLE: "Краткосрочная дебиторская задолженность покупателей",
    VAT: "НДС к уплате",
    PAYABLE: "Краткосрочная кредиторская задолженность поставщикам",
    ADVANCES: "Краткосрочные авансы полученные",
    OPENING: "Нераспределённая прибыль (входящие остатки)",
    REVENUE: "Доход от реализации услуг",
    FX_GAIN: "Доходы от курсовой разницы",
    EXPENSE: "Административные расходы",
    FX_LOSS: "Расходы по курсовой разнице",
}

# Legacy name kept for optional modules written against the first prototype:
# unallocated receipts are now parked in advances received.
SUSPENSE = ADVANCES
