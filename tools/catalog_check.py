"""Verify every worked example in docs/error_catalog.md against data/answers/*/truth.json.

Run from repository root:
    python tools/catalog_check.py

Uses only the Python standard library. Checks recalculated `expected_amount_minor`
(using manifest minor_units exponent scaling per policy.fx.formula), `type`,
`currency`, `affected_ids`, `tolerance_minor`, ledger account numbers, and that
each catalog minimal `evidence_set` is accepted by `truth.json`.
"""
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"


def round_half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def fx_convert(amount_minor: int, rate: Decimal, source_ccy: str, manifest: dict) -> int:
    minor_units = manifest["minor_units"]
    exp_diff = minor_units[manifest["base_currency"]] - minor_units[source_ccy]
    return round_half_up(Decimal(amount_minor) * rate * (Decimal(10) ** exp_diff))


def load_open_episode(split: str, episode_id: str):
    ep_dir = DATA / "episodes" / split / episode_id
    ans_dir = DATA / "answers" / split / episode_id
    manifest = json.loads((ep_dir / "manifest.json").read_text(encoding="utf-8"))
    invoices = {
        r["invoice_id"]: r
        for line in (ep_dir / "invoices.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
        for r in (json.loads(line),)
    }
    payments = {
        r["payment_id"]: r
        for line in (ep_dir / "payments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
        for r in (json.loads(line),)
    }
    with (ep_dir / "ledger.csv").open(encoding="utf-8", newline="") as f:
        ledger = {r["row_id"]: r for r in csv.DictReader(f)}
    with (ep_dir / "fx.csv").open(encoding="utf-8", newline="") as f:
        fx = {r["fx_id"]: r for r in csv.DictReader(f)}
    truth = json.loads((ans_dir / "truth.json").read_text(encoding="utf-8"))
    issues = {item["issue_id"]: item for item in truth["issues"]}
    return manifest, invoices, payments, ledger, fx, issues


def _verify_entry(label, ep_ref, iss, *, expected_type, expected_ccy, expected_tol, expected_affected,
                  recalc_minor, catalog_evidence_set):
    errors = []
    if recalc_minor != iss["expected_amount_minor"]:
        errors.append(f"amount {recalc_minor} != {iss['expected_amount_minor']}")
    if iss["type"] != expected_type:
        errors.append(f"type {iss['type']} != {expected_type}")
    if iss["currency"] != expected_ccy:
        errors.append(f"ccy {iss['currency']} != {expected_ccy}")
    if iss["tolerance_minor"] != expected_tol:
        errors.append(f"tol {iss['tolerance_minor']} != {expected_tol}")
    if set(iss["affected_ids"]) != set(expected_affected):
        errors.append("affected_ids mismatch")
    ev_sets = [set(s) for s in iss["evidence_sets"]]
    if set(catalog_evidence_set) not in ev_sets:
        errors.append("catalog evidence_set not in truth.json")
    return (label, ep_ref, iss["issue_id"], recalc_minor, iss["expected_amount_minor"], iss["currency"], errors)


def check_examples():
    checks = []

    # 1. duplicate_invoice
    _, invs, _, ledger, _, issues = load_open_episode("dev", "episode_01aad47789ce2be8074fd17d")
    assert ledger["row_1eb0bc35b537e6e41d77f6a3"]["account"] == "6010"
    assert ledger["row_72443f581ac7688e71ffb976"]["account"] == "6010"
    iss = issues["issue_4f3dc449d29c9c442763fc1b"]
    calc = invs["invoice_529e2d2d8baa08e4b57d7862"]["total_minor"]
    checks.append(_verify_entry(
        "1. duplicate_invoice", "dev/episode_01aad47789ce2be8074fd17d", iss,
        expected_type="duplicate_invoice", expected_ccy="KZT", expected_tol=0,
        expected_affected=["invoice_51badc2da8ebe82b85e25f7a", "invoice_529e2d2d8baa08e4b57d7862"],
        recalc_minor=calc,
        catalog_evidence_set=[
            "doc_88c7297216807319f91a503f", "doc_2c3e42e735517524b3b9b4c0",
            "row_1eb0bc35b537e6e41d77f6a3", "row_72443f581ac7688e71ffb976",
        ],
    ))

    # 2. payment_allocation
    _, _, _, ledger, _, issues = load_open_episode("dev", "episode_3de70bbf58543fdff8d3cc5f")
    assert ledger["row_1df54e2fd354da82e6dd45d6"]["account"] == "1030"
    assert ledger["row_66972cd6341bef025ffaa0f9"]["account"] == "3510"
    iss = issues["issue_012cea7ac7b2c82b9a8a2e81"]
    calc = int(ledger["row_66972cd6341bef025ffaa0f9"]["credit_minor"])
    checks.append(_verify_entry(
        "2. payment_allocation", "dev/episode_3de70bbf58543fdff8d3cc5f", iss,
        expected_type="payment_allocation", expected_ccy="KZT", expected_tol=0,
        expected_affected=["payment_90381551eae689cf11683322"],
        recalc_minor=calc,
        catalog_evidence_set=[
            "doc_7e74a7c2ceb37a050aef89cc", "doc_aedb6bb611a386f0ed7d547e",
            "row_1df54e2fd354da82e6dd45d6", "row_66972cd6341bef025ffaa0f9",
        ],
    ))

    # 3. fx_rate
    man, _, pays, ledger, fx, issues = load_open_episode("dev", "episode_27f78b89946ce866415ffc44")
    assert ledger["row_a8f685cbeeb197d54726eaf9"]["account"] == "1030"
    iss = issues["issue_fd6a0618a1be97c86de9d973"]
    pay = pays["payment_95c1057137016537a47da7bb"]
    rate = Decimal(fx["fx_df74cf23bc21695ee3374475"]["rate"])
    expected_kzt = fx_convert(pay["amount_minor"], rate, pay["currency"], man)
    booked_kzt = int(ledger["row_a8f685cbeeb197d54726eaf9"]["debit_minor"])
    checks.append(_verify_entry(
        "3. fx_rate", "dev/episode_27f78b89946ce866415ffc44", iss,
        expected_type="fx_rate", expected_ccy="KZT", expected_tol=2,
        expected_affected=["payment_95c1057137016537a47da7bb"],
        recalc_minor=abs(booked_kzt - expected_kzt),
        catalog_evidence_set=[
            "doc_24ec8bd3390b293cd9bb4e9b", "fx_df74cf23bc21695ee3374475", "row_a8f685cbeeb197d54726eaf9",
        ],
    ))

    # 4. period_cutoff
    _, _, _, ledger, _, issues = load_open_episode("dev", "episode_29dde7eeab65f74d1c15ccfd")
    assert ledger["row_23dfad85630ced0d7bb6059b"]["account"] == "6010"
    iss = issues["issue_459037a87dfa98c0e6798c85"]
    calc = int(ledger["row_23dfad85630ced0d7bb6059b"]["credit_minor"])
    checks.append(_verify_entry(
        "4. period_cutoff", "dev/episode_29dde7eeab65f74d1c15ccfd", iss,
        expected_type="period_cutoff", expected_ccy="KZT", expected_tol=0,
        expected_affected=["invoice_d1b7f821f519fc1b5da84cf5"],
        recalc_minor=calc,
        catalog_evidence_set=["doc_7779d8c4cab812459f9806af", "row_23dfad85630ced0d7bb6059b"],
    ))

    # 5. vat_on_top
    _, invs, _, ledger, _, issues = load_open_episode("dev", "episode_3de70bbf58543fdff8d3cc5f")
    assert ledger["row_af47473f33073cd0dc274db5"]["account"] == "3130"
    iss = issues["issue_205fd190a5d63727851fd650"]
    inv = invs["invoice_6fc55b518d199ef2abad44ca"]
    calc = sum(li["total_minor"] - li["quantity"] * li["unit_price_minor"] for li in inv["line_items"])
    checks.append(_verify_entry(
        "5. vat_on_top", "dev/episode_3de70bbf58543fdff8d3cc5f", iss,
        expected_type="vat_on_top", expected_ccy="KZT", expected_tol=0,
        expected_affected=["invoice_6fc55b518d199ef2abad44ca"],
        recalc_minor=calc,
        catalog_evidence_set=["doc_3092b7406a9ac620ecf8f9fe", "row_af47473f33073cd0dc274db5"],
    ))

    # 6. uncollected_service
    _, invs, _, ledger, _, issues = load_open_episode("dev", "episode_3de70bbf58543fdff8d3cc5f")
    assert ledger["row_31b63bf3b7f6e9155ba4bda8"]["account"] == "1210"
    iss = issues["issue_6a4dc840b60781c38534fa08"]
    calc = invs["invoice_1fa6d4f32aa5e0fbcf513a72"]["total_minor"]
    checks.append(_verify_entry(
        "6. uncollected_service", "dev/episode_3de70bbf58543fdff8d3cc5f", iss,
        expected_type="uncollected_service", expected_ccy="KZT", expected_tol=0,
        expected_affected=["invoice_1fa6d4f32aa5e0fbcf513a72"],
        recalc_minor=calc,
        catalog_evidence_set=[
            "doc_f30beea96fe277872e0b3e94", "doc_b811a51a52b5c81a192b969f", "row_31b63bf3b7f6e9155ba4bda8",
        ],
    ))

    # 7. IX-A fx_boundary_payment
    man, _, pays, ledger, fx, issues = load_open_episode("dev", "episode_4712fc0a77b544a7f2ca0caf")
    assert ledger["row_4a5227efc98bca705d6fd18d"]["account"] == "1030"
    booked_ix_a = int(ledger["row_4a5227efc98bca705d6fd18d"]["debit_minor"])
    iss_cut = issues["issue_32c3c888107f1da349ce50d6"]
    checks.append(_verify_entry(
        "7a. fx_boundary_payment (cutoff)", "dev/episode_4712fc0a77b544a7f2ca0caf", iss_cut,
        expected_type="period_cutoff", expected_ccy="KZT", expected_tol=0,
        expected_affected=["payment_505175d11fe112e4215978c5"],
        recalc_minor=booked_ix_a,
        catalog_evidence_set=["doc_4b3bdf472353d01f328e8bee", "row_4a5227efc98bca705d6fd18d"],
    ))

    iss_fx = issues["issue_b1802c8570e6dbd48e6bb060"]
    pay_ix_a = pays["payment_505175d11fe112e4215978c5"]
    rate_ix_a = Decimal(fx["fx_5e1199d9d94f6658325b7dd2"]["rate"])
    expected_ix_a = fx_convert(pay_ix_a["amount_minor"], rate_ix_a, pay_ix_a["currency"], man)
    checks.append(_verify_entry(
        "7b. fx_boundary_payment (fx_rate)", "dev/episode_4712fc0a77b544a7f2ca0caf", iss_fx,
        expected_type="fx_rate", expected_ccy="KZT", expected_tol=2,
        expected_affected=["payment_505175d11fe112e4215978c5"],
        recalc_minor=abs(booked_ix_a - expected_ix_a),
        catalog_evidence_set=[
            "doc_4b3bdf472353d01f328e8bee", "fx_5e1199d9d94f6658325b7dd2", "row_4a5227efc98bca705d6fd18d",
        ],
    ))

    # 8. IX-B fx_partial_allocation
    man, _, pays, ledger, fx, issues = load_open_episode("dev", "episode_a76adfd653b5ca2e26e4ccfe")
    assert ledger["row_9691fffa3595d54f9584ed18"]["account"] == "1030"
    assert ledger["row_0cbd8b148597b3b261122153"]["account"] == "3510"
    iss_alloc = issues["issue_11990b97537e55857e1f0db5"]
    calc_alloc = int(ledger["row_0cbd8b148597b3b261122153"]["credit_minor"])
    checks.append(_verify_entry(
        "8a. fx_partial_allocation (alloc)", "dev/episode_a76adfd653b5ca2e26e4ccfe", iss_alloc,
        expected_type="payment_allocation", expected_ccy="KZT", expected_tol=0,
        expected_affected=["payment_13bcdb915d3dbca8d2150e2e"],
        recalc_minor=calc_alloc,
        catalog_evidence_set=[
            "doc_87433413fd064c0f36c54c08", "doc_1c4127f6353e8664a0827d92",
            "row_9691fffa3595d54f9584ed18", "row_0cbd8b148597b3b261122153",
        ],
    ))

    iss_fx2 = issues["issue_5df2c583043446771bc58510"]
    pay_ix_b = pays["payment_13bcdb915d3dbca8d2150e2e"]
    rate_ix_b = Decimal(fx["fx_64006b2e265fe7e2aeab1715"]["rate"])
    expected_ix_b = fx_convert(pay_ix_b["amount_minor"], rate_ix_b, pay_ix_b["currency"], man)
    booked_ix_b = int(ledger["row_9691fffa3595d54f9584ed18"]["debit_minor"])
    checks.append(_verify_entry(
        "8b. fx_partial_allocation (fx_rate)", "dev/episode_a76adfd653b5ca2e26e4ccfe", iss_fx2,
        expected_type="fx_rate", expected_ccy="KZT", expected_tol=2,
        expected_affected=["payment_13bcdb915d3dbca8d2150e2e"],
        recalc_minor=abs(booked_ix_b - expected_ix_b),
        catalog_evidence_set=[
            "doc_87433413fd064c0f36c54c08", "fx_64006b2e265fe7e2aeab1715", "row_9691fffa3595d54f9584ed18",
        ],
    ))

    return checks


def main():
    checks = check_examples()
    failures = 0
    print(f"{'example':36s} {'episode':36s} {'recalc':>10s} {'truth':>10s} {'ccy':>4s}  status")
    for label, ep, _issue_id, recalc, expected, ccy, errors in checks:
        ok = not errors
        if not ok:
            failures += 1
        status = "OK" if ok else "FAIL: " + "; ".join(errors)
        print(f"{label:36s} {ep:36s} {recalc:10d} {expected:10d} {ccy:>4s}  {status}")
    print(f"checked {len(checks)} catalog entries: {len(checks) - failures}/{len(checks)} matched")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
