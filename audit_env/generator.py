"""Trusted generation: hidden world -> clean books -> recorded errors -> public files."""
import re
from copy import deepcopy
from dataclasses import replace
from random import Random

from . import books, keys, render, scenarios, world as world_module
from .contracts import MAX_ISSUES, Config, Episode, Issue

MIN_ROWS, MAX_ROWS = 200, 2000
MIN_DOCUMENTS, MAX_DOCUMENTS = 20, 200
PUBLIC_ID = re.compile(r"^(episode|row|doc|invoice|payment|entity|issue|fx)_[0-9a-f]{24}$")


def generate(seed, config, key=None):
    """Same seed, config and key always give byte-identical files and truth."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    config = Config.from_dict(config)
    key = keys.resolve(config.keyspace) if key is None else key
    semantic = config.semantic_json()
    extra = 0
    for _ in range(6):
        world = world_module.build(keys.stream(key, "world", seed, semantic), config, extra_filler=extra)
        clean = books.post(world)
        if len(clean) >= MIN_ROWS + 12:
            break
        extra += (MIN_ROWS + 12 - len(clean)) // 2 + 4
    overrides, specs = scenarios.inject(keys.stream(key, "errors", seed, semantic), world, config)
    rows = books.post(world, overrides)
    resolved = [scenarios.resolve(spec, world, overrides, rows) for spec in specs]
    episode = render.present(world, clean, rows, overrides, resolved, key=key, seed=seed, config=config)
    check_bounds(episode)
    return episode


def check_bounds(episode):
    counts = episode.manifest["counts"]
    if not MIN_ROWS <= counts["ledger"] <= MAX_ROWS:
        raise AssertionError(f"ledger size {counts['ledger']} outside [{MIN_ROWS}, {MAX_ROWS}]")
    if not MIN_DOCUMENTS <= episode.manifest["document_count"] <= MAX_DOCUMENTS:
        raise AssertionError(f"document count {episode.manifest['document_count']} outside bounds")
    if len(episode.issues) > MAX_ISSUES:
        raise AssertionError("too many issues")
    for issue in episode.issues:
        for proof in issue.evidence_sets:
            if not 1 <= len(proof) <= 10:
                raise AssertionError("evidence set must hold 1..10 ids")


def rename_ids(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [rename_ids(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(rename_ids(item, mapping) for item in value)
    if isinstance(value, dict):
        return {mapping.get(key, key): rename_ids(item, mapping) for key, item in value.items()}
    return value


def collect_ids(value, found):
    if isinstance(value, str):
        if PUBLIC_ID.match(value):
            found.add(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            collect_ids(item, found)
    elif isinstance(value, dict):
        for key, item in value.items():
            collect_ids(key, found)
            collect_ids(item, found)


def transform(episode, seed):
    """Paired check: rename every opaque ID and reshuffle every table; the answer is preserved."""
    rng = Random(seed)
    found = set()
    for part in (episode.manifest, episode.tables, episode.documents, episode.provenance):
        collect_ids(part, found)
    for issue in episode.issues:
        collect_ids([issue.issue_id, issue.affected_ids, issue.evidence_sets, issue.relevant_ids], found)
    mapping = {old: f"{old.split('_', 1)[0]}_{rng.getrandbits(96):024x}" for old in sorted(found)}
    tables = rename_ids(deepcopy(episode.tables), mapping)
    for rows in tables.values():
        rng.shuffle(rows)
    issues = tuple(Issue(**{name: rename_ids(getattr(issue, name), mapping) for name in Issue.__dataclass_fields__})
                   for issue in episode.issues)
    issues = tuple(replace(issue, affected_ids=tuple(sorted(issue.affected_ids)),
                           relevant_ids=tuple(sorted(issue.relevant_ids))) for issue in issues)
    provenance = rename_ids(deepcopy(episode.provenance), mapping) | {"transform_seed": seed}
    return Episode(rename_ids(deepcopy(episode.manifest), mapping), deepcopy(episode.policy), tables,
                   issues, provenance, rename_ids(deepcopy(episode.documents), mapping))
