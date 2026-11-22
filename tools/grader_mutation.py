"""Mutation testing harness for audit_env/scoring.py (standard library only).

Generates up to 60 single-point mutants of `audit_env/scoring.py`:
- condition removal in `matches()` (one clause dropped at a time);
- operator flips (`==` <-> `!=`, `<=` <-> `<`, `and` <-> `or`, `+` <-> `-`);
- constant/limit perturbations (`0`/`1`/`2`, `STEP_COST`, `MAX_FINDINGS`, `MAX_EVIDENCE`, `MAX_EXPLANATION`).

Each mutant is evaluated in an isolated temporary directory (source repo is never
modified) using `python -m unittest tests.test_scoring tests.test_gaps
tests.test_attacks tests.test_brief_criteria -q` across 2 worker processes.

Run from repository root:
    python tools/grader_mutation.py
"""
import ast
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORING_REL = Path("audit_env") / "scoring.py"
TEST_MODULES = (
    "tests.test_scoring",
    "tests.test_gaps",
    "tests.test_attacks",
    "tests.test_brief_criteria",
    "tests.test_scoring_boundaries",
)
MAX_MUTANTS = 60
WORKERS = 2


def _docstring_lines(source: str) -> set[int]:
    """Return 1-based line numbers belonging to module/class/function docstrings."""
    lines: set[int] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                start = node.body[0].lineno
                end = node.body[0].end_lineno or start
                lines.update(range(start, end + 1))
    return lines


def generate_mutants(source: str, max_mutants: int = MAX_MUTANTS):
    """Return list of (id, line_no, description, mutated_source) tuples."""
    lines = source.splitlines(keepends=True)
    doc_lines = _docstring_lines(source)
    mutants = []
    seen_sources = set()

    def add(line_no: int, desc: str, new_lines: list[str]):
        if len(mutants) >= max_mutants:
            return
        text = "".join(new_lines)
        if text == source or text in seen_sources:
            return
        seen_sources.add(text)
        mutants.append((len(mutants) + 1, line_no, desc, text))

    # 1. Remove each individual condition inside matches() (lines between def matches and next def)
    in_matches = False
    for idx, line in enumerate(lines):
        if line.startswith("def matches("):
            in_matches = True
            continue
        if in_matches and line.startswith("def "):
            in_matches = False
        if in_matches and (idx + 1) not in doc_lines:
            stripped = line.strip()
            if stripped.startswith(("finding[", "and ")):
                new_lines = list(lines)
                indent = line[: len(line) - len(line.lstrip())]
                if stripped.startswith("and "):
                    new_lines[idx] = f"{indent}and True\n"
                    add(idx + 1, f"matches: drop `{stripped}`", new_lines)
                else:
                    new_lines[idx] = f"{indent}True\n"
                    add(idx + 1, f"matches: drop `{stripped}`", new_lines)

    # 2. Limit & constant perturbations in scoring.py (core official grader before MaterialityScore)
    limit_replacements = [
        ("STEP_COST = 0.002", "STEP_COST = 0.0", "STEP_COST 0.002 -> 0.0"),
        ("STEP_COST = 0.002", "STEP_COST = 0.02", "STEP_COST 0.002 -> 0.02"),
        ("MAX_FINDINGS", "(MAX_FINDINGS + 1)", "MAX_FINDINGS -> +1"),
        ("MAX_EVIDENCE", "(MAX_EVIDENCE + 1)", "MAX_EVIDENCE -> +1"),
        ("MAX_EXPLANATION", "(MAX_EXPLANATION + 1)", "MAX_EXPLANATION -> +1"),
        ("MAX_AMOUNT_MINOR", "(MAX_AMOUNT_MINOR * 10)", "MAX_AMOUNT_MINOR -> *10"),
        ("1 <= len(value) <= limit", "0 <= len(value) <= limit", "valid_ids min 1 -> 0"),
        ("1 <= len(value) <= limit", "1 <= len(value) <= limit + 1", "valid_ids max -> +1"),
        ("len(finding[\"currency\"]) == 3", "len(finding[\"currency\"]) >= 2", "currency len == 3 -> >= 2"),
        ("2 * tp / denominator if denominator else 1.0", "2 * tp / denominator if denominator else 0.0", "empty F1 1.0 -> 0.0"),
        ("2 * f1 - 1 - STEP_COST * steps", "2 * f1 - 1 + STEP_COST * steps", "reward - STEP_COST -> + STEP_COST"),
        ("2 * f1 - 1 - STEP_COST * steps", "2 * f1 - 2 - STEP_COST * steps", "reward - 1 -> - 2"),
        ("min(1.0, max(-1.0, 2 * f1 - 1 - STEP_COST * steps))", "2 * f1 - 1 - STEP_COST * steps", "remove reward clip"),
        ("Score(0, 0, len(issues), 0.0, 0.0, 0.0, -1.0", "Score(0, 0, len(issues), 0.0, 0.0, 0.0, 0.0", "reject reward -1.0 -> 0.0"),
        ("len(findings) - tp", "len(findings) + tp", "fp: - tp -> + tp"),
        ("len(issues) - tp", "len(issues) + tp", "fn: - tp -> + tp"),
    ]
    for idx, line in enumerate(lines):
        if line.startswith("class MaterialityScore"):
            break
        if (idx + 1) in doc_lines or line.lstrip().startswith(("#", '"', "'", "from ", "import ")):
            continue
        for old, new, desc in limit_replacements:
            if old in line:
                new_lines = list(lines)
                new_lines[idx] = line.replace(old, new, 1)
                add(idx + 1, desc, new_lines)

    # 3. Single-operator mutations (== <-> !=, <= <-> <, and <-> or, + <-> -) on core scoring logic
    op_rules = [
        (" == ", " != ", "== -> !="),
        (" != ", " == ", "!= -> =="),
        (" <= ", " < ", "<= -> <"),
        (" < ", " <= ", "< -> <="),
        (" and ", " or ", "and -> or"),
        (" or ", " and ", "or -> and"),
        (" + ", " - ", "+ -> -"),
        (" - ", " + ", "- -> +"),
    ]
    for idx, line in enumerate(lines):
        if len(mutants) >= max_mutants:
            break
        if line.startswith("class MaterialityScore"):
            break
        stripped = line.strip()
        if (
            (idx + 1) in doc_lines
            or not stripped
            or stripped.startswith(("#", '"', "'", "def ", "from ", "import ", "@"))
        ):
            continue
        for old, new, desc in op_rules:
            if old in line:
                new_lines = list(lines)
                new_lines[idx] = line.replace(old, new, 1)
                add(idx + 1, f"{desc}: `{stripped[:52]}`", new_lines)
                if len(mutants) >= max_mutants:
                    break

    return mutants[:max_mutants]


def _prepare_sandbox(tmp_root: Path):
    """Create a lightweight copy of the repo in tmp_root without touching REPO_ROOT."""
    shutil.copytree(
        REPO_ROOT / "audit_env",
        tmp_root / "audit_env",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        REPO_ROOT / "tests",
        tmp_root / "tests",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("data", "configs", "schemas", "models", "pyproject.toml"):
        src = REPO_ROOT / name
        dst = tmp_root / name
        if not src.exists():
            continue
        try:
            os.symlink(src, dst, target_is_directory=src.is_dir())
        except OSError:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


def _eval_mutant(args):
    mid, line_no, desc, mutated_src = args
    with tempfile.TemporaryDirectory(prefix=f"mut_{mid}_") as tmp:
        tmp_root = Path(tmp)
        _prepare_sandbox(tmp_root)
        (tmp_root / SCORING_REL).write_text(mutated_src, encoding="utf-8")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "unittest", *TEST_MODULES, "-q"],
                cwd=str(tmp_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            killed = proc.returncode != 0
        except subprocess.TimeoutExpired:
            killed = True
        return (mid, line_no, desc, killed)


def main():
    started = time.perf_counter()
    source = (REPO_ROOT / SCORING_REL).read_text(encoding="utf-8")
    mutants = generate_mutants(source, max_mutants=MAX_MUTANTS)
    print(f"Grader mutation testing on {SCORING_REL} ({len(mutants)} mutants, {WORKERS} workers)")
    print(f"{'id':>3s} {'line':>4s} {'status':8s}  mutation")

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(_eval_mutant, mutants))

    killed_count = 0
    survived = []
    for mid, line_no, desc, killed in results:
        if killed:
            killed_count += 1
            status = "KILLED"
        else:
            status = "SURVIVED"
            survived.append((mid, line_no, desc))
        print(f"{mid:3d} {line_no:4d} {status:8s}  {desc}")

    elapsed = time.perf_counter() - started
    total = len(results)
    pct = (100.0 * killed_count / total) if total else 0.0
    print(f"\nmutation score = {killed_count}/{total} ({pct:.1f}%) in {elapsed:.1f} s")
    if survived:
        print(f"surviving mutants ({len(survived)}):")
        for mid, line_no, desc in survived:
            print(f"  #{mid} L{line_no}: {desc}")
    else:
        print("surviving mutants: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
