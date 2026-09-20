from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from . import config

log = logging.getLogger(__name__)

_H1 = re.compile(r"^#\s+(.+?)\s*$")
_RULE = re.compile(r"^\s*(\d+)\.\s+(.*)$")

@dataclass(frozen=True)
class Rule:
    doc: str
    index: int
    text: str

@dataclass(frozen=True)
class PolicyDoc:
    doc: str
    title: str
    body: str
    rules: tuple[Rule, ...]

@dataclass(frozen=True)
class Policy:
    fingerprint: str
    prefix: str
    facts: Mapping[str, Mapping[str, int]]
    docs: Mapping[str, PolicyDoc]

@dataclass(frozen=True)
class _FactSpec:
    doc: str
    key: str
    rule: int
    pattern: str
    group: int = 1
    rupees: bool = False

_FACT_SPECS: tuple[_FactSpec, ...] = (
    _FactSpec("returns", "window_days", 1, r"within\s+(\d+)\s+calendar days"),
    _FactSpec("damaged_goods", "window_days", 1, r"within\s+(\d+)\s+calendar days"),
    _FactSpec("damaged_goods", "photo_threshold_inr", 3, r"above\s+₹\s*([\d,]+)", rupees=True),
    _FactSpec("wrong_item", "window_days", 1, r"within\s+(\d+)\s+calendar days"),
    _FactSpec("defective_products", "window_days", 1, r"within\s+(\d+)\s+calendar days"),
    _FactSpec(
        "defective_products",
        "evidence_threshold_inr",
        2,
        r"above\s+₹\s*([\d,]+)",
        rupees=True,
    ),
    _FactSpec("shipping", "expected_days", 1, r"within\s+(\d+)\s+calendar days"),
    _FactSpec("shipping", "wait_from", 2, r"(\d+)\s+or\s+(\d+)\s+days", group=1),
    _FactSpec("shipping", "wait_to", 2, r"(\d+)\s+or\s+(\d+)\s+days", group=2),
    _FactSpec("shipping", "investigate_from", 3, r"(\d+)\s+to\s+(\d+)\s+days", group=1),
    _FactSpec("shipping", "investigate_to", 3, r"(\d+)\s+to\s+(\d+)\s+days", group=2),
    _FactSpec("shipping", "offer_after_days", 4, r"more than\s+(\d+)\s+days"),
)

def _parse_doc(path: Path) -> PolicyDoc:
    text = path.read_text(encoding="utf-8")
    stem = path.stem
    title = ""
    rules: list[Rule] = []
    index: int | None = None
    pending: list[str] = []

    def flush() -> None:
        nonlocal index, pending
        if index is not None:
            rules.append(Rule(stem, index, " ".join(pending).strip()))
        index, pending = None, []

    for raw in text.splitlines():
        heading = _H1.match(raw)
        if heading and not title:
            title = heading.group(1).strip()
            continue
        rule = _RULE.match(raw)
        if rule:
            flush()
            index = int(rule.group(1))
            pending = [rule.group(2).strip()]
            continue
        if index is not None and raw.strip():
            pending.append(raw.strip())
    flush()

    return PolicyDoc(doc=stem, title=title, body=text, rules=tuple(rules))

def _doc_paths() -> list[Path]:
    if not config.KB_DIR.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {config.KB_DIR}")
    paths = sorted(config.KB_DIR.glob("*.md"), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No policy documents (*.md) found in {config.KB_DIR}")
    return paths

def load_documents() -> dict[str, PolicyDoc]:
    return {path.stem: _parse_doc(path) for path in _doc_paths()}

def build_prefix(docs: Mapping[str, PolicyDoc] | None = None) -> str:
    docs = docs if docs is not None else load_documents()
    sections = [
        f"=== POLICY DOCUMENT: {doc.doc}.md ===\n{doc.body.strip()}\n"
        for doc in docs.values()
    ]
    return "\n".join(sections)

def _derive_facts(docs: Mapping[str, PolicyDoc]) -> dict[str, dict[str, int]]:
    facts: dict[str, dict[str, int]] = {}
    problems: list[str] = []
    for spec in _FACT_SPECS:
        doc = docs.get(spec.doc)
        if doc is None:
            problems.append(f"{spec.doc}.{spec.key}: document not found")
            continue
        rule = next((r for r in doc.rules if r.index == spec.rule), None)
        if rule is None:
            problems.append(f"{spec.doc}.{spec.key}: rule {spec.rule} not found")
            continue
        match = re.search(spec.pattern, rule.text, flags=re.IGNORECASE)
        if not match:
            problems.append(
                f"{spec.doc}.{spec.key}: pattern not matched in rule {spec.rule}: {rule.text!r}"
            )
            continue
        raw = match.group(spec.group).replace(",", "")
        facts.setdefault(spec.doc, {})[spec.key] = int(raw)

    if problems:
        raise ValueError(
            "Could not read the following policy facts from the knowledge base at "
            f"{config.KB_DIR}. A policy document has been reworded:\n  - "
            + "\n  - ".join(problems)
        )
    return facts

def fingerprint() -> str:
    digest = hashlib.sha256()
    for path in _doc_paths():
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

_policy: Policy | None = None

def get_policy() -> Policy:
    global _policy
    current = fingerprint()
    if _policy is None or _policy.fingerprint != current:
        existing = _policy is not None
        docs = load_documents()
        _policy = Policy(
            fingerprint=current,
            prefix=build_prefix(docs),
            facts=_derive_facts(docs),
            docs=docs,
        )
        log.info(
            "%s policy context from %d documents (fingerprint %s)",
            "Rebuilt" if existing else "Built",
            len(docs),
            current[:12],
        )
    return _policy

def get_prefix() -> str:
    return get_policy().prefix

def get_facts() -> Mapping[str, Mapping[str, int]]:
    return get_policy().facts

def invalidate() -> None:
    global _policy
    _policy = None

def iter_rules() -> Iterator[Rule]:
    for doc in get_policy().docs.values():
        yield from doc.rules
