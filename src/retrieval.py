import hashlib
import json
import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from . import cache, config, llm

log = logging.getLogger(__name__)

DEFAULT_K = 3

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str
    title: str
    rule: int
    text: str
    references: tuple[str, ...] = ()

    @property
    def searchable(self) -> str:
        return f"{self.doc} policy, rule {self.rule}: {self.text}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.searchable.encode("utf-8")).hexdigest()

def cross_references(
    text: str, docs: Mapping[str, "cache.PolicyDoc"], self_doc: str
) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(
        sorted(
            stem
            for stem, doc in docs.items()
            if stem != self_doc and doc.title and doc.title.lower() in lowered
        )
    )

def load_chunks() -> list[Chunk]:
    policy = cache.get_policy()
    docs = policy.docs
    return [
        Chunk(
            chunk_id=f"{rule.doc}#{rule.index}",
            doc=rule.doc,
            title=docs[rule.doc].title,
            rule=rule.index,
            text=rule.text,
            references=cross_references(rule.text, docs, rule.doc),
        )
        for rule in cache.iter_rules()
    ]

def _read_vector_cache() -> dict[str, list[float]]:
    path = config.EMBEDDINGS_PATH
    if not path.exists():
        return {}
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("ignoring unreadable vector cache at %s", path)
        return {}
    return cached if isinstance(cached, dict) else {}

def _write_vector_cache(cached: Mapping[str, Sequence[float]]) -> None:
    path = config.EMBEDDINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cached), encoding="utf-8")

def embed_chunks(chunks: Sequence[Chunk] | None = None) -> tuple[list[Chunk], list[list[float]]]:
    chunks = load_chunks() if chunks is None else list(chunks)
    cached = _read_vector_cache()

    missing = [chunk for chunk in chunks if chunk.digest not in cached]
    if missing:
        log.info("embedding %d chunk(s), %d reused from cache", len(missing), len(chunks) - len(missing))
        for chunk, vector in zip(missing, llm.embed([chunk.searchable for chunk in missing])):
            cached[chunk.digest] = vector
        _write_vector_cache(cached)

    return chunks, [cached[chunk.digest] for chunk in chunks]

def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)

def retrieve(query: str, k: int = DEFAULT_K) -> list[Chunk]:
    chunks, vectors = embed_chunks()
    if not chunks:
        return []
    query_vector = llm.embed([query])[0]
    ranked = sorted(
        zip(chunks, vectors), key=lambda pair: cosine(query_vector, pair[1]), reverse=True
    )
    return [chunk for chunk, _ in ranked[: max(0, k)]]

def retrieve_docs(query: str, k: int = DEFAULT_K) -> list[str]:
    seen: list[str] = []
    for chunk in retrieve(query, k):
        if chunk.doc not in seen:
            seen.append(chunk.doc)
    return seen

if __name__ == "__main__":
    chunks = load_chunks()
    assert chunks, "expected at least one chunk from the knowledge base"
    assert all(chunk.text.strip() for chunk in chunks), "every chunk carries rule text"
    assert any(chunk.references for chunk in chunks), "cross-policy references are recorded"
    for chunk in chunks:
        assert all(reference != chunk.doc for reference in chunk.references)
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    print(f"{len(chunks)} chunks: " + ", ".join(chunk.chunk_id for chunk in chunks))
