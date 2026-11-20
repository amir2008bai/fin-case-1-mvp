"""Hidden company history: customers, sales invoices (acts), receipts, other operations.

Nothing here is an audit error. Scenario modules later change bookkeeping
decisions or add a duplicate act; books.py turns events into ledger rows and
render.py turns events into public files with opaque IDs.
"""
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

from .contracts import ALLOCATION, CUTOFF, DUPLICATE, FX_BOUNDARY, FX_PARTIAL, FX_RATE, UNCOLLECTED, VAT_ON_TOP
from .domain import BASE_CURRENCY, FOREIGN_CURRENCIES
from .fx import build_rates, convert

STATEMENT_TAIL_DAYS = 3  # the bank statement runs a few days past the period end
FX_LEAD_DAYS = 45
FX_TAIL_DAYS = 7
MIN_FX_GAP_MINOR = 200  # an injected FX discrepancy is at least 100× the tolerance

CATALOG = (
    # code, description, VAT applies, monthly service, prices in KZT
    ("SUB", "Подписка на облачное ПО", False, True, (4_600, 5_000, 8_000, 10_000, 20_000)),
    ("SUP", "Техническое сопровождение ПО", True, True, (10_000, 15_000, 20_000, 42_000, 48_000)),
    ("IMP", "Внедрение и настройка учётной системы", True, False, (60_000, 80_000, 96_000, 116_000, 150_000)),
    ("TRN", "Обучение пользователей", True, False, (15_000, 20_000, 36_000)),
    ("DEV", "Доработка конфигурации", True, False, (40_000, 80_000, 120_000)),
)
GOODS = (
    # code, description, unit prices in KZT, cost share of the net price
    ("G-KKM", "Онлайн-ККМ (кассовый аппарат)", (85_000, 98_000, 120_000), 70),
    ("G-POS", "POS-терминал", (110_000, 135_000), 75),
    ("G-SCN", "Сканер штрихкода", (18_000, 25_000, 32_000), 62),
    ("G-PRN", "Принтер чеков", (39_000, 45_000, 52_000), 66),
    ("G-LBL", "Принтер этикеток", (55_000, 68_000), 68),
    ("G-TAP", "Термолента для ККМ (рулон)", (350, 420, 500), 55),
    ("G-DRW", "Денежный ящик", (21_000, 26_000), 60),
)
FOREIGN_SERVICE = ("EXP", "Сопровождение ПО (экспорт услуг)")
FOREIGN_PRICES = {"USD": (150, 300, 450, 600, 900, 1_200), "RUB": (15_000, 30_000, 45_000, 90_000, 120_000)}
COUNTRY_OF = {"USD": "KG", "RUB": "RU"}
COUNTRY_OFFSET = {"KZ": None, "KG": "+06:00", "RU": "+03:00"}
NAME_A = ("Алтын", "Нур", "Сары", "Жана", "Ак", "Кок", "Бай", "Тау", "Дала", "Шыгыс", "Орда",
          "Есиль", "Арна", "Сункар", "Береке", "Кайнар", "Самал", "Достык", "Аян", "Кулан")
NAME_B = ("Сервис", "Строй", "Трейд", "Учёт", "Групп", "Логистик", "Финанс", "Дом", "Плюс",
          "Систем", "Консалт", "Медиа", "Агро", "Техно", "Сауда", "Курылыс")
FORMS = {"KZ": ("ТОО", "ТОО", "ИП", "ОСИ", "АО"), "KG": ("ОсОО",), "RU": ("ООО",)}
MANAGERS = ("Ахметова Д.", "Сейткали А.", "Иванова О.", "Нурланов Е.", "Ким С.")
SUPPLIERS = ("ТОО «Облачный хостинг»", "ТОО «Связь Плюс»", "ИП «Офис Сервис»", "ТОО «Рекрутинг Онлайн»",
             "ТОО «Кассовое Оборудование»")
SHOWN_OFFSETS = ("-05:00", "+00:00", "+03:00", "+05:00", "+06:00", "+09:00")


def parse_offset(text):
    sign = 1 if text[0] == "+" else -1
    return timezone(sign * timedelta(hours=int(text[1:3]), minutes=int(text[4:6])))


@dataclass(frozen=True)
class Line:
    line_no: int
    service_code: str
    description: str
    quantity: int
    unit_price_minor: int
    vat_rate: str | None
    period_start: date
    period_end: date
    cost_minor: int = 0          # goods only: cost of the shipped quantity
    vat_on_top: bool = False     # an error: VAT added on top of the agreed price

    @property
    def base_minor(self):
        return self.quantity * self.unit_price_minor

    @property
    def total_minor(self):
        return self.base_minor + (self.vat_minor if self.vat_on_top else 0)

    @property
    def vat_minor(self):
        """VAT is included in the price and computed per line, rounded half up."""
        if self.vat_rate is None:
            return 0
        rate = Decimal(self.vat_rate)
        if self.vat_on_top:
            return int((Decimal(self.base_minor) * rate).quantize(Decimal(1), ROUND_HALF_UP))
        return int((Decimal(self.base_minor) * rate / (1 + rate)).quantize(Decimal(1), ROUND_HALF_UP))

    @property
    def net_minor(self):
        return self.total_minor - self.vat_minor


@dataclass(frozen=True)
class Act:
    key: str
    customer: str
    currency: str
    issued_at: datetime
    due_days: int
    lines: tuple[Line, ...]
    manager: str
    contract: str
    origin: str
    kind: str = "service"  # "service" (act of completed works) or "goods" (waybill)

    @property
    def service_start(self):
        return min(line.period_start for line in self.lines)

    @property
    def service_end(self):
        return max(line.period_end for line in self.lines)

    @property
    def total_minor(self):
        return sum(line.total_minor for line in self.lines)

    @property
    def due_at(self):
        return self.issued_at + timedelta(days=self.due_days)


@dataclass(frozen=True)
class Payment:
    key: str
    customer: str
    currency: str
    amount_minor: int  # receipts are positive, refunds to customers negative
    instant: datetime
    shown_offset: str
    refs: tuple[str, ...] | None
    kind: str
    match_act: str | None = None
    match_at: datetime | None = None


@dataclass(frozen=True)
class Filler:
    key: str
    at: datetime
    debit: str
    credit: str
    amount_minor: int
    entity: str | None


@dataclass(frozen=True)
class Customer:
    key: str
    name: str
    country: str
    currency: str


@dataclass
class World:
    period_start: datetime
    period_end: datetime
    tz: timezone
    tz_text: str
    company: str
    customers: dict
    acts: dict
    payments: dict
    opening_advances: dict
    opening_bank_minor: int
    fillers: list
    suppliers: dict
    rates: dict
    fx_first: date
    fx_last: date
    candidates: dict = field(default_factory=dict)
    summary: Counter = field(default_factory=Counter)
    opening_inventory_minor: int = 0

    def local_date(self, instant):
        return instant.astimezone(self.tz).date()

    def in_period(self, instant):
        return self.period_start <= instant < self.period_end

    def recognized_at(self, act):
        return datetime.combine(act.service_end, time(0, 0), self.tz)

    def rate(self, day, currency):
        return self.rates[(day, currency)]


class Builder:
    def __init__(self, rng, config):
        self.rng = rng
        self.config = config
        self.on = config.distractors
        self.tz_text = config.timezone
        self.tz = parse_offset(config.timezone)
        month = config.month or rng.randint(1, 12)
        self.start = datetime(2026, month, 1, tzinfo=self.tz)
        self.end = (datetime(2027, 1, 1, tzinfo=self.tz) if month == 12
                    else datetime(2026, month + 1, 1, tzinfo=self.tz))
        self.first = self.start.date()
        self.last = (self.end - timedelta(days=1)).date()
        self.window_end = self.end + timedelta(days=STATEMENT_TAIL_DAYS)
        self.customers, self.acts, self.payments = {}, {}, {}
        self.fillers, self.opening_advances = [], {}
        self.names, self.busy, self.reserved = set(), set(), set()
        self.paid = Counter()
        self.settled = set()
        self.starts, self.nexts = set(), set()
        self.summary = Counter()
        self.candidates = {"dup": [], "alloc": [], "fx": [], "cut": [], FX_BOUNDARY: [], FX_PARTIAL: [],
                           "vat": [], "uncollected": []}
        self.rates = build_rates(rng, FOREIGN_CURRENCIES, self.first - timedelta(days=FX_LEAD_DAYS),
                                 self.last + timedelta(days=FX_TAIL_DAYS))
        self.opening_bank = 0

    # ---- helpers -------------------------------------------------------
    def chance(self, percent):
        return self.rng.randrange(100) < percent

    def day_between(self, low, high):
        span = (high - low).days
        return low + timedelta(days=self.rng.randrange(span + 1)) if span >= 0 else low

    def moment(self, day, low_hour=9, high_hour=19):
        return datetime.combine(day, time(self.rng.randrange(low_hour, high_hour + 1),
                                          self.rng.randrange(60), self.rng.randrange(60)), self.tz)

    def name(self, country):
        for _ in range(500):
            form = self.rng.choice(FORMS[country])
            text = f"{form} «{self.rng.choice(NAME_A)} {self.rng.choice(NAME_B)}»"
            if text not in self.names:
                self.names.add(text)
                return text
        raise RuntimeError("name space exhausted")

    def new_customer(self, currency):
        key = f"C{len(self.customers) + 1:03d}"
        country = "KZ" if currency == BASE_CURRENCY else COUNTRY_OF[currency]
        self.customers[key] = Customer(key, self.name(country), country, currency)
        return key

    def goods_lines(self, service_end, count):
        lines, codes = [], set()
        for _ in range(count):
            code, text, prices, share = self.rng.choice(GOODS)
            if code in codes:
                continue
            codes.add(code)
            quantity = self.rng.randrange(10, 101, 10) if code == "G-TAP" else self.rng.choice((1, 1, 1, 2, 3, 5))
            price = self.rng.choice(prices) * 100
            vat = "0.16" if self.config.vat else None
            probe = Line(0, code, text, quantity, price, vat, service_end, service_end)
            cost = probe.net_minor * share // 100
            lines.append((code, text, quantity, price, vat, service_end, service_end, cost))
        return lines

    def lines_for(self, customer, service_end, count, kind="service", code=None):
        currency = self.customers[customer].currency
        if kind == "goods":
            return self.goods_lines(service_end, count)
        if code is not None:
            item = next(entry for entry in CATALOG if entry[0] == code)
            vat = "0.16" if item[2] and self.config.vat else None
            return [(code, item[1], 1, self.rng.choice(item[4]) * 100, vat, service_end - timedelta(days=29), service_end)]
        if currency != BASE_CURRENCY:
            price = self.rng.choice(FOREIGN_PRICES[currency]) * 100
            begin = service_end - timedelta(days=self.rng.choice((0, 6, 13, 29)))
            return [(FOREIGN_SERVICE[0], FOREIGN_SERVICE[1], 1, price, None, begin, service_end)]
        lines, codes = [], set()
        for _ in range(count):
            code, text, taxable, monthly, prices = self.rng.choice(CATALOG)
            if code in codes:
                continue
            codes.add(code)
            begin = service_end - timedelta(days=29 if monthly else self.rng.randrange(0, 15))
            vat = "0.16" if taxable and self.config.vat else None
            lines.append((code, text, 1, self.rng.choice(prices) * 100, vat, begin, service_end))
        return lines

    def new_act(self, customer, origin, service_end, issued_at=None, lines=1, max_lag=7, kind="service",
                code=None, due_days=None):
        for _ in range(60):
            items = tuple(Line(index, *values) for index, values in
                          enumerate(self.lines_for(customer, service_end, lines, kind, code), 1))
            marks = {(customer, line.service_code, line.period_start, line.period_end) for line in items}
            starts = {(customer, line.service_code, line.period_start) for line in items}
            nexts = {(customer, line.service_code, line.period_end + timedelta(days=1)) for line in items}
            # never a back-to-back renewal by accident: those are reserved for the uncollected scenario
            if not marks & self.busy and not starts & self.nexts and not nexts & self.starts:
                self.starts |= starts
                self.nexts |= nexts
                break
            service_end += timedelta(days=-1 if self.rng.randrange(2) else 1)
        else:
            raise RuntimeError("could not place a unique act")
        self.busy |= marks
        if issued_at is None:
            lag = min(max_lag, self.rng.choice((0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 5, 7)))
            issued_at = self.moment(service_end + timedelta(days=lag), 10, 18)
        key = f"A{len(self.acts) + 1:03d}"
        if self.chance(60):
            signed = date(self.rng.choice((2023, 2024, 2025)), self.rng.randint(1, 12), self.rng.randint(1, 28))
            contract = f"Договор № {self.rng.randrange(1, 400)} от {signed.isoformat()}"
        else:
            contract = "Публичная оферта"
        act = Act(key, customer, self.customers[customer].currency, issued_at,
                  due_days or self.rng.choice((10, 14, 30)), items, self.rng.choice(MANAGERS), contract, origin, kind)
        self.acts[key] = act
        self.summary[f"acts.{origin}"] += 1
        self.summary[f"documents.{kind}"] += 1
        return act

    def shown(self, customer):
        country = self.customers[customer].country
        if COUNTRY_OFFSET[country]:
            return COUNTRY_OFFSET[country]
        return "+00:00" if self.chance(30) else self.tz_text

    def pay(self, customer, amount, instant, refs, kind, shown=None, match=None):
        if instant is None or not (self.start <= instant < self.window_end):
            return None
        key = f"P{len(self.payments) + 1:03d}"
        match_act, match_at = match if match else (None, None)
        payment = Payment(key, customer, self.customers[customer].currency, amount, instant,
                          shown or self.shown(customer), tuple(refs) if refs is not None else None,
                          kind, match_act, match_at)
        self.payments[key] = payment
        for ref in refs or ():
            self.paid[ref] += 1
            if instant < self.end:
                self.settled.add(ref)
        if match_act:
            self.paid[match_act] += 1
            if match_at < self.end:
                self.settled.add(match_act)
        self.summary[f"payments.{kind}"] += 1
        return payment

    def recognized(self, act):
        return datetime.combine(act.service_end, time(0), self.tz)

    def after(self, act, low_days=0, high_days=15):
        base = max(self.recognized(act), act.issued_at, self.start)
        day = base.date() + timedelta(days=self.rng.randrange(low_days, high_days + 1))
        instant = self.moment(day)
        return instant if instant > base else base + timedelta(hours=self.rng.randrange(1, 6))

    # ---- composition ---------------------------------------------------
    def build(self):
        config = self.config
        errors, mix = config.errors, config.interactions
        needs_foreign = errors[FX_RATE] or any(mix.values()) or (errors[DUPLICATE] and config.foreign_percent)
        foreign_count = max(round(config.customers * config.foreign_percent / 100), 2 if needs_foreign else 0)
        foreign_count = min(foreign_count, config.customers - 3)
        self.domestic = [self.new_customer(BASE_CURRENCY) for _ in range(config.customers - foreign_count)]
        self.foreign = [self.new_customer(self.rng.choice(FOREIGN_CURRENCIES)) for _ in range(foreign_count)]

        self.reserve_targets()
        dedicated = len(self.acts)
        prior = round(config.acts * 0.10) if self.on["opening_balances"] else 0
        late = max(1, round(config.acts * 0.05)) if self.on["late_acts"] else 0
        upcoming = max(1, round(config.acts * 0.05)) if self.on["prepayments"] else 0
        regular = max(6, config.acts - dedicated - prior - late - upcoming)
        foreign_regular = round(regular * config.foreign_percent / 100) if self.foreign else 0
        # goods_percent is the share of ALL invoices of the episode that are waybills for goods
        planned = dedicated + regular + prior + late + upcoming
        dedicated_goods = sum(act.kind == "goods" for act in self.acts.values())
        goods = min(regular - foreign_regular, max(0, round(planned * config.goods_percent / 100) - dedicated_goods))
        for index in range(regular):
            pool = self.foreign if index < foreign_regular else self.domestic
            kind = "goods" if foreign_regular <= index < foreign_regular + goods else "service"
            self.regular_act(self.rng.choice(pool), kind=kind)
        for _ in range(prior):
            self.prior_act(self.rng.choice(self.domestic))
        for _ in range(late):
            act = self.late_act(self.rng.choice(self.domestic))
            if self.chance(60):
                self.pay(act.customer, act.total_minor, self.after(act, 0, 12), [act.key], "full")
        for _ in range(upcoming):
            act = self.next_act(self.rng.choice(self.domestic))
            if self.chance(60):
                day = self.day_between(act.issued_at.date(), self.last)
                instant = max(self.moment(day), act.issued_at + timedelta(minutes=30))
                if instant < self.end:
                    self.pay(act.customer, act.total_minor, instant, [act.key], "prepayment_next")
        self.multi_act_payments()
        self.opening()
        self.boundaries()
        self.collect_overdue_renewed()
        self.filler()
        goods_cost = sum(line.cost_minor for act in self.acts.values() for line in act.lines)
        world = World(self.start, self.end, self.tz, self.tz_text, self.name("KZ"), self.customers,
                      self.acts, self.payments, self.opening_advances, self.opening_bank,
                      self.fillers, {f"S{index}": name for index, name in enumerate(SUPPLIERS)},
                      self.rates, self.first - timedelta(days=FX_LEAD_DAYS),
                      self.last + timedelta(days=FX_TAIL_DAYS), self.candidates, self.summary)
        world.opening_inventory_minor = 2 * goods_cost if goods_cost else 0
        return world

    def collect_overdue_renewed(self):
        """Clean books never renew a service for a customer whose earlier invoice is overdue and unpaid."""
        by_customer = {}
        for act in self.acts.values():
            by_customer.setdefault(act.customer, []).append(act)
        for act in list(self.acts.values()):
            if (act.key in self.reserved or act.key in self.settled or act.currency != BASE_CURRENCY
                    or act.due_at >= self.end or self.recognized(act) >= self.end):
                continue
            follows = {(line.service_code, line.period_end + timedelta(days=1)) for line in act.lines}
            renewed = any(other.key != act.key and other.issued_at > act.due_at
                          and follows & {(line.service_code, line.period_start) for line in other.lines}
                          for other in by_customer[act.customer])
            if renewed:
                base = max(self.recognized(act), act.issued_at, self.start) + timedelta(hours=2)
                if base < self.end:
                    self.pay(act.customer, act.total_minor, base, [act.key], "full")

    def regular_act(self, customer, *, lines=None, target=False, kind="service"):
        currency = self.customers[customer].currency
        count = lines or (2 if currency == BASE_CURRENCY and self.chance(18) else 1)
        if target:
            service_end = self.day_between(self.first, self.last - timedelta(days=7))
            act = self.new_act(customer, "regular", service_end, lines=count, max_lag=2, kind=kind)
            self.reserved.add(act.key)
            return act
        act = self.new_act(customer, "regular", self.day_between(self.first, self.last), lines=count, kind=kind)
        self.receipts_for(act)
        return act

    def receipts_for(self, act):
        on, roll = self.on, self.rng.randrange(100)
        domestic = act.currency == BASE_CURRENCY
        if roll < 50:
            self.pay(act.customer, act.total_minor, self.after(act), [act.key], "full")
        elif roll < 62 and on["split_payments"]:
            first = act.total_minor * self.rng.randrange(30, 71) // 100
            one = self.pay(act.customer, first, self.after(act, 0, 8), [act.key], "split")
            if one:
                second = one.instant + timedelta(days=self.rng.randrange(0, 10), hours=self.rng.randrange(1, 9))
                self.pay(act.customer, act.total_minor - first, second, [act.key], "split")
        elif roll < 70 and on["overpayments"] and domestic and act.total_minor <= 2_000_000:
            payment = self.pay(act.customer, act.total_minor * 10, self.after(act, 0, 10), [act.key], "overpayment")
            if payment and on["refunds"] and self.chance(60):
                when = payment.instant + timedelta(days=self.rng.randrange(1, 8))
                if when < self.end:
                    self.pay(act.customer, -act.total_minor * 9, when, None, "refund")
        elif roll < 78 and on["unreferenced_payments"] and domestic:
            instant = self.after(act, 0, 10)
            when = instant + timedelta(days=self.rng.randrange(1, 6))
            match = (act.key, when) if self.chance(65) and when < self.end else None
            self.pay(act.customer, act.total_minor, instant, None, "unreferenced", match=match)
        elif roll < 86 and on["prepayments"] and domestic:
            self.prepaid_act_receipt(act)
        # otherwise the act stays open at period end

    def prepaid_act_receipt(self, act):
        """Receipt cites the invoice before the service ends: an advance, offset at recognition."""
        low = max(act.service_start, self.first, act.issued_at.date())
        high = act.service_end - timedelta(days=1)
        if high < low:
            self.pay(act.customer, act.total_minor, self.after(act), [act.key], "full")
            return
        instant = max(self.moment(self.day_between(low, high)), act.issued_at + timedelta(minutes=30))
        if self.start <= instant < self.recognized(act):
            self.pay(act.customer, act.total_minor, instant, [act.key], "prepayment")

    def prior_act(self, customer):
        service_end = self.day_between(self.first - timedelta(days=40), self.first - timedelta(days=12))
        act = self.new_act(customer, "prior_open", service_end, max_lag=3)
        if self.chance(70):
            self.pay(customer, act.total_minor, self.moment(self.day_between(self.first, self.last)),
                     [act.key], "full")
        return act

    def late_act(self, customer):
        service_end = self.day_between(self.first - timedelta(days=10), self.first - timedelta(days=1))
        issued = self.moment(self.day_between(self.first, self.first + timedelta(days=9)), 10, 18)
        return self.new_act(customer, "late", service_end, issued_at=issued)

    def next_act(self, customer):
        issued = self.moment(self.day_between(self.last - timedelta(days=3), self.last), 10, 18)
        service_end = self.day_between(self.last + timedelta(days=5), self.last + timedelta(days=30))
        return self.new_act(customer, "next_period", service_end, issued_at=issued)

    # ---- dedicated targets for configured errors -----------------------
    def reserve_targets(self):
        errors, mix = self.config.errors, self.config.interactions
        domestic, foreign = self.domestic, self.foreign
        for _ in range(errors[DUPLICATE]):
            pool = foreign if foreign and self.chance(20) else domestic
            act = self.regular_act(self.rng.choice(pool), lines=self.rng.choice((1, 1, 2)), target=True)
            if self.chance(50):
                self.pay(act.customer, act.total_minor, self.after(act, 0, 4), [act.key], "full")
            self.candidates["dup"].append(act.key)
        for _ in range(errors[ALLOCATION]):
            act = self.regular_act(self.rng.choice(domestic), target=True)
            payment = self.full_in_period(act)
            self.candidates["alloc"].append((act.key, payment.key))
        for _ in range(errors[FX_RATE]):
            act = self.regular_act(self.rng.choice(foreign), target=True)
            if self.chance(50):
                if self.chance(50):
                    self.pay(act.customer, act.total_minor, self.after(act, 0, 4), [act.key], "full")
                self.candidates["fx"].append(("act", act.key, None))
            else:
                payment = self.full_in_period(act, distinct_rate=True)
                self.candidates["fx"].append(("payment", act.key, payment.key))
        for _ in range(errors[CUTOFF]):
            variant = self.rng.choice(("act_prior", "act_next", "payment"))
            if variant == "act_prior":
                act = self.late_act(self.rng.choice(domestic))
                self.reserved.add(act.key)
                self.candidates["cut"].append(("act_prior", act.key, None))
            elif variant == "act_next":
                act = self.next_act(self.rng.choice(domestic))
                self.reserved.add(act.key)
                self.candidates["cut"].append(("act_next", act.key, None))
            else:
                act = self.regular_act(self.rng.choice(domestic), target=True)
                payment = self.after_end(act, act.total_minor)
                self.candidates["cut"].append(("payment", act.key, payment.key))
        for _ in range(mix[FX_BOUNDARY]):
            for _attempt in range(30):
                act = self.regular_act(self.rng.choice(foreign), target=True)
                part = act.total_minor * self.rng.randrange(40, 71) // 100
                if self.gap(part, act.currency, self.end.date(), self.last) >= MIN_FX_GAP_MINOR:
                    break
            else:
                raise RuntimeError("no foreign invoice large enough for a boundary receipt")
            payment = self.after_end(act, part, kind="partial")
            self.candidates[FX_BOUNDARY].append((act.key, payment.key))
        for _ in range(mix[FX_PARTIAL]):
            act = self.regular_act(self.rng.choice(foreign), target=True)
            payment = self.full_in_period(act, distinct_rate=True)
            self.candidates[FX_PARTIAL].append((act.key, payment.key))
        for _ in range(errors[VAT_ON_TOP]):
            kind = "goods" if self.config.goods_percent and self.chance(50) else "service"
            act = self.regular_act(self.rng.choice(domestic), target=True, kind=kind)
            payment = self.full_in_period(act)  # the customer pays the agreed price
            self.candidates["vat"].append((act.key, payment.key))
        for _ in range(errors[UNCOLLECTED]):
            customer = self.new_customer(BASE_CURRENCY)  # dedicated: no other documents of this customer
            service_end = self.day_between(self.first, self.first + timedelta(days=6))
            act = self.new_act(customer, "regular", service_end, lines=1, max_lag=1,
                               code=self.rng.choice(("SUB", "SUP")), due_days=10)
            self.reserved.add(act.key)
            self.candidates["uncollected"].append(act.key)

    def gap(self, amount, currency, first_day, second_day):
        return abs(convert(amount, currency, self.rates[(first_day, currency)])
                   - convert(amount, currency, self.rates[(second_day, currency)]))

    def full_in_period(self, act, distinct_rate=False):
        for _ in range(60):
            instant = self.after(act, 1, 5)
            local = instant.astimezone(self.tz).date()
            if not (max(self.start, act.issued_at) < instant < self.end):
                continue
            if distinct_rate and (local == act.service_end or self.gap(
                    act.total_minor, act.currency, local, act.service_end) < MIN_FX_GAP_MINOR):
                continue
            return self.pay(act.customer, act.total_minor, instant, [act.key], "full")
        raise RuntimeError("no slot for an in-period receipt")

    def after_end(self, act, amount, kind="full"):
        """Receipt just after the period end, shown in an offset whose wall date is still inside."""
        own = self.tz.utcoffset(None)
        minutes = self.rng.randrange(5, 150)
        instant = self.end + timedelta(minutes=minutes)
        options = [text for text in SHOWN_OFFSETS
                   if parse_offset(text).utcoffset(None) < own - timedelta(minutes=minutes)]
        return self.pay(act.customer, amount, instant, [act.key], kind, shown=self.rng.choice(options))

    # ---- balances, boundaries and other operations ---------------------
    def multi_act_payments(self):
        if not self.on["multi_act_payments"]:
            return
        by_customer = {}
        for act in self.acts.values():
            if (act.origin == "regular" and act.currency == BASE_CURRENCY and self.paid[act.key] == 0
                    and act.key not in self.reserved and act.service_end <= self.last - timedelta(days=2)):
                by_customer.setdefault(act.customer, []).append(act)
        made = 0
        for customer in sorted(by_customer):
            acts = by_customer[customer]
            if len(acts) < 2 or made >= 3:
                continue
            first, second = acts[0], acts[1]
            base = max(self.recognized(first), self.recognized(second), first.issued_at, second.issued_at)
            instant = base + timedelta(days=self.rng.randrange(0, 4), hours=self.rng.randrange(1, 10))
            if instant < self.end and self.pay(customer, first.total_minor + second.total_minor, instant,
                                               [first.key, second.key], "multi_act"):
                made += 1

    def opening(self):
        self.opening_bank = self.rng.randrange(2_000_000, 9_000_000) * 100
        if not self.on["opening_balances"]:
            return
        acts_of = {}
        for act in self.acts.values():
            acts_of.setdefault(act.customer, []).append(act)
        eligible = [key for key in self.domestic
                    if len(acts_of.get(key, ())) == 1 and acts_of[key][0].origin == "regular"
                    and acts_of[key][0].key not in self.reserved]
        for customer in eligible[: max(1, len(eligible) // 5)]:
            act = acts_of[customer][0]
            self.opening_advances[customer] = act.total_minor * self.rng.randrange(20, 101) // 100
            self.summary["opening_advances"] += 1
        lonely = [key for key in self.domestic if key not in acts_of]
        if self.on["refunds"] and lonely:
            customer = self.rng.choice(lonely)
            amount = self.rng.choice((5_000, 10_000, 20_000)) * 100
            self.opening_advances[customer] = amount
            self.summary["opening_advances"] += 1
            self.pay(customer, -amount, self.moment(self.day_between(self.first, self.last)), None, "refund")

    def boundaries(self):
        if not self.on["boundary_timestamps"]:
            return
        own = self.tz.utcoffset(None)
        open_acts = [act for act in self.acts.values()
                     if act.origin == "regular" and act.currency == BASE_CURRENCY and self.paid[act.key] == 0
                     and act.key not in self.reserved and act.service_end <= self.last - timedelta(days=1)
                     and act.issued_at < self.end - timedelta(days=1)]
        self.rng.shuffle(open_acts)
        for act in open_acts[:2]:  # inside the period, but the shown wall date is the next day
            minutes = self.rng.randrange(5, 90)
            instant = self.end - timedelta(minutes=minutes)
            options = [text for text in SHOWN_OFFSETS
                       if parse_offset(text).utcoffset(None) >= own + timedelta(minutes=minutes)]
            if options:
                self.pay(act.customer, act.total_minor, instant, [act.key], "boundary_inside",
                         shown=self.rng.choice(options))
        for act in open_acts[2:3]:  # after the period end, shown in the company offset
            instant = self.end + timedelta(hours=self.rng.randrange(1, 40))
            self.pay(act.customer, act.total_minor, instant, [act.key], "after_period", shown=self.tz_text)
        prior = [act for act in self.acts.values() if act.origin == "prior_open" and self.paid[act.key] == 0]
        for act in prior[:1]:  # inside the period, but the shown wall date is in the previous month
            minutes = self.rng.randrange(5, 120)
            instant = self.start + timedelta(minutes=minutes)
            options = [text for text in SHOWN_OFFSETS
                       if parse_offset(text).utcoffset(None) < own - timedelta(minutes=minutes)]
            self.pay(act.customer, act.total_minor, instant, [act.key], "boundary_start",
                     shown=self.rng.choice(options))

    def filler(self):
        for index in range(self.config.filler):
            at = self.moment(self.day_between(self.first, self.last), 8, 20)
            kind = self.rng.randrange(3)
            supplier = f"S{self.rng.randrange(len(SUPPLIERS))}"
            if kind == 0:
                item = Filler(f"F{index:03d}", at, "7210", "1030", self.rng.randrange(300, 2_500) * 100, None)
            elif kind == 1:
                item = Filler(f"F{index:03d}", at, "7210", "3310", self.rng.randrange(5_000, 300_000) * 100, supplier)
            else:
                item = Filler(f"F{index:03d}", at, "3310", "1030", self.rng.randrange(5_000, 300_000) * 100, supplier)
            self.fillers.append(item)
        self.summary["filler"] = self.config.filler
        goods_acts = sum(act.kind == "goods" for act in self.acts.values())
        for index in range(max(2, goods_acts // 4) if goods_acts else 0):  # supplier deliveries of goods
            at = self.moment(self.day_between(self.first, self.last), 9, 17)
            net = self.rng.randrange(200_000, 2_000_000) * 100
            supplier = f"S{len(SUPPLIERS) - 1}"
            self.fillers.append(Filler(f"G{index:03d}", at, "1330", "3310", net, supplier))
            if self.config.vat:
                self.fillers.append(Filler(f"G{index:03d}V", at, "1420", "3310", net * 16 // 100, supplier))
        self.summary["goods_deliveries"] = max(2, goods_acts // 4) if goods_acts else 0


def build(rng, config, extra_filler=0):
    if extra_filler:
        config = replace(config, filler=min(400, config.filler + extra_filler))
    return Builder(rng, config).build()
