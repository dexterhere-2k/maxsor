# Development notes: engineering decisions & agent log

This document records the decisions, trade-offs, and AI-agent workflow behind the support-ticket
decision assistant. Every figure came from a run of this code. Where a number is missing, the
reason is stated instead of an estimate.

---

## 1. Engineering philosophy: minimum that works, measured

The brief grades simplicity and the ability to explain decisions. Three rules were applied
throughout:

- **Standard library first.** `sqlite3` with `row_factory` gives dictionary rows, transactions,
  and zero dependencies in ~220 lines. No ORM, no session lifecycle, no migrations.
- **Every threshold comes from the documents.** No rupee value or day window is typed anywhere in
  `src/`. `cache._derive_facts` parses each one out of a specific numbered rule of a specific
  policy, and a document that stops yielding one is a startup failure, not a default.
  `python -m src.decision` asserts that `2000` appears nowhere in that module.
- **Every figure is produced or labelled.** The spec's rule "never reuse a number" is enforced by
  the M-table in section 6: measured, or `not measured - <reason>`.

What was rejected up front: SQLAlchemy, LangChain/LlamaIndex, a hosted vector DB, a multi-agent
architecture. Not because they are bad, but because a 2,967-byte corpus does not need them.

---

## 2. Why CAG serves and retrieval only benchmarks

### 1. Token math

- Corpus: 6 documents, 29 numbered rules, 2,967 bytes.
- Measured served prompt: median **1,090 tokens** (policies + ticket + instructions), against a
  Flash window measured in hundreds of thousands.
- Context utilisation is a rounding error. There is no prompt bloat to avoid.

### 2. The severance problem, and what the comparison actually showed

Retrieval is fully implemented (`retrieval.py`: H1-and-rule chunking, cross-reference edges,
content-hash-cached embeddings, cosine ranking) and never serves a request. `api.py` and
`decision.py` do not import it, and the test suite checks that in a fresh interpreter.

Rather than argue about chunk boundaries in the abstract, `compare.py` ran 11 tickets (the 5
supplied cases plus 6 labelled rows) through both pipelines:

```
M1 fidelity, served (CAG)          10/11 (91%)
M1 fidelity, retrieval (mini-RAG)  10/11 (91%)
rows where the pipelines disagree   2 of 11
M5 prompt tokens served            median 1090
M5 prompt tokens retrieval         median 455
```

A tie, but the failures were not the same failure:

- **CSV002, retrieval wrong.** A damaged order at 4 days and 1,799 rupees, label
  `APPROVE_REFUND_OR_REPLACEMENT`. Top-3 retrieval missed `damaged_goods#3`, the rule carrying
  the 2,000-rupee threshold, so the model answered `REQUEST_DEFECT_EVIDENCE`. This is the
  severance problem happening for real: the deciding clause was not in the retrieved set.
- **CSV003, CAG wrong.** An unopened non-food return at 10 days, label `APPROVE_RETURN`.
  Full-corpus CAG answered `REJECT_OUTSIDE_WINDOW` - the 7-day reading from the damaged-goods
  policy, not the 14-day returns window. Restricted to three `returns.md` chunks, the model
  answered correctly. Reading: the full corpus put the wrong policy's window within reach.

The honest conclusion is that neither arrangement is strictly safer at this size, which is a
better result than the table alone. His 25% latency figure is not reproduced here because the
latency rows in that run included the deliberate pacing wait; the report says so.

### 3. Caching does not apply and is not claimed

Gemini's minimum cacheable prompt is 2,048 tokens; the prefix is below the floor. Nothing reads
`total_cached_tokens`, and no cached-token figure exists anywhere in the schema.

---

## 3. Backend and database decisions

- **sqlite3 over SQLAlchemy.** Raw SQL, ~220 lines, zero dependencies.
- **WAL placement.** `PRAGMA journal_mode=WAL` runs once at `init_db()`, not per connection;
  `foreign_keys=ON` runs per connection. Verified by `python -m src.database`.
- **Authorization in the SQL.** `get_ticket(user_id, ticket_id)` filters on `user_id`, and a
  foreign ticket returns 404, indistinguishable from a missing one, so ids cannot be probed.
  Pinned by the tenant test.
- **bcrypt with a per-hash salt; PyJWT with a 24-hour expiry.** The JWT secret is never shipped
  with a default: unset, the app generates an ephemeral one and warns that tokens will not
  survive a restart. `load_dotenv` is scoped to this project's `.env` after it was caught walking
  up the tree and reading a `.env` from outside the project.

### Schema additions beyond the minimum, and why

| Addition | Reason |
|---|---|
| `decisions.path` | makes every stored decision attributable: model vs rule engine |
| `decisions.prompt_tokens` | makes the token figure measured, not estimated |
| `tickets.order_value_inr` ... `order_status` | a decision cannot be re-derived from the message alone |
| CHECK constraints on `action`, `confidence`, `path` | the vocabulary is enforced twice, once by Pydantic and once by SQLite |

---

## 4. The decision pipeline and its failure handling

### 1. Two paths, always attributable

`path` is `cag` when a key is configured and the model answered, `fallback` otherwise. A configured model that fails is a 503 and stores nothing - it is never silently replaced by
the rule engine, because a fallback answering in place of the model makes every accuracy figure
unreadable. A reachable model that answers unusably is different: one corrective retry, then a
forced `NEEDS_MORE_INFORMATION` with empty sources, still attributed to `cag`.

### 2. The validation chain

Every model answer passes four gates before it is stored:

1. **Schema.** The `Decision` schema goes in as `response_format`, and the reply is parsed and
   Pydantic-validated regardless of what the provider did with the schema. A bad answer gets one
   retry with the rejection reason appended to the prompt.
2. **Citations.** Every cited source must appear in the context actually supplied. Unverifiable
   citations are retried once, then dropped.
3. **Context signal.** The signal is the fraction of the answered action's relevant fields that
   are known. Below 0.5, no concrete action survives; the answer becomes
   `NEEDS_MORE_INFORMATION`.
4. **Confidence cap.** For a concrete action, the model's self-reported confidence is capped at
   `0.50 + 0.45 * signal`, so a full-facts answer tops out at 0.95 and the model can talk itself
   down but never up. A `NEEDS_MORE_INFORMATION` answer is scored by `_confidence` instead, the
   same formula the rule engine uses: base 0.50 plus 0.30 times the routed signal. (Routing it
   through the cap once made every request-for-details answer read 0.95: the action has no entry
   in `_ACTION_FIELDS`, the empty field list scored a signal of 1.0, and the cap went to its
   maximum for the one answer that by definition knows the least.)

The signal and the cap follow the **answered action's** fields, not a keyword router's guess.
That ordering was itself a bug once: the original guard scored against the router's guess, so
unrecognised wording superseded a model answer that had read the facts correctly, and a
fully-specified ticket capped at 0.50 while an unmatchable one capped at 0.95.

### 3. Facts written in prose

`got my items spilled, purchased 4 days ago for 1300` was initially answered
`NEEDS_MORE_INFORMATION` claiming the date and value were missing. They were in the message.
Two fixes:

- `_extract_facts` reads rupee amounts (`rs 1300`, `1300 rupees`, `for 1300`, `paid 19999`) and
  `N days ago` into fields the submitter left blank. Typed values are never overridden.
- The prompt says a null in the order facts is an instruction to the form-filler, not to the
  model: if the customer states the fact in prose, read it there.

A keyword-list patch was also applied (`spilled`, `cracked`, ...) and then reverted. Growing
the vocabulary one real query at a time never converges, and it is the same fit-to-the-input move
as tuning a prompt against a test case. The fallback's limit is stated in its own reason text
instead: it needs the usual terms or the structured fields.

### 4. The fallback engine

The deterministic engine reads only the parsed facts and the historical wording, agrees with
214/214 historical labels and all 16 boundary cases, and records itself honestly as `fallback`.
The 214/214 is not a claim of intelligence: the label file is 30 templates where each message
maps to exactly one action and the wording already gives away the threshold (section 6).

---

## 5. Frontend decisions

- **HTTP only.** No direct DB access; every call carries `Authorization: Bearer <jwt>`.
- **Session state, knowingly.** The token lives in `st.session_state`, which survives tab
  switches but resets on a hard reload. Fine for a demo; production would persist it in a
  cookie. This trade-off is stated rather than hidden.
- **Message first.** The required input is the ticket message. Order details sit in a collapsed
  section, because an agent often does not have the order value to hand. A message-only submit
  returns `NEEDS_MORE_INFORMATION` naming the missing fact, and the agent adds it and resubmits.
- **Failure is legible.** A 503 renders as "the model could not be reached, so no decision was
  made or stored", not a silent offline answer.

---

## 6. What the evaluation can and cannot show

### The label file cannot separate reasoning from matching

`data/tickets.csv` is 214 rows that are 30 distinct messages, each mapping to exactly one action,
no exceptions. 24 of the 30 templates repeat, covering 208 rows, and a leave-one-out
nearest-neighbour lookup on exact message text scores **208/208** - the lookup the brief forbids,
printed next to the model's score for that reason. Worse, every template is threshold-flat: the
message that says "expensive" is the above-2,000 branch, and its 1,999-rupee sibling says
something else. No pipeline has to evaluate `order_value_inr > 2000` to score well on it.

So `Accuracy: 100%` on the five supplied cases is a smoke test (one case is 20 points), and the
file that looks like an evaluation set measures template recall.

### The boundary probe is the measurement that discriminates

`data/boundary_probe.json` holds 16 author-written pairs straddling every threshold with wording
held constant. The rule engine passes 16/16. The live model passes **15/16**; the failure is B09,
a return at exactly 14 days, read as outside the window. Three of the four day-window policies
also state their exclusion sentence ("more than N days ... not eligible"); `returns.md` is the
only one that does not, and it is the only boundary the model gets wrong. The probe found a model
error and a documentation gap in the same place.

It is left unfixed on purpose. A prompt line to flip B09 would be fitting the prompt to the probe.
If fixed, it should be a general reading rule or a tightened sentence in the source policy.

### The metrics ledger

| # | Metric | Result |
|---|---|---|
| M1 | Answer fidelity | 10/11 both pipelines on 11 rows; baseline 208/208 |
| M2 | Retrieval recall | not measured - no supplied governing-document label |
| M3 | Severance cases | partially observed - 2 of 11 rows disagreed, both diagnosed in section 2 |
| M4 | Latency | p50 6.31s served / 5.80s restricted / 5.14s retrieval leg, with a 4s pace in effect |
| M5 | Token cost | median 1,090 served / 455 restricted; cost per 1,000 not measured, no price applied |
| M6 | Corpus threshold | 2,967 bytes, 29 rules, median 1,090 tokens; window share not measured |
| M7 | Version correctness | not measured live - one version per policy; selection is covered by `python -m src.cache` |

Version-aware loading (front matter with `version`, `effective_from`, `supersedes`) is
implemented and tested: a future-dated document never enters the prefix, and a newer file with
the same H1 title supersedes the older one. Building it exposed that `_parse_doc` parsed front
matter but never attached it, so the filtering code was correct and did nothing.

---

## 7. Agent usage log

The project was built with an AI coding agent in the loop. What it proposed, and what happened:

**Accepted.**

- CAG as the serving path with retrieval kept as a benchmark - deleting `retrieval.py` would have
  removed the only comparison that produced the section-2 findings.
- 503 on a configured-model outage, storing nothing; retry-then-refuse for reachable-but-wrong.
- Contents-only fingerprint against the spec's "contents and mtimes" wording: identical text must
  not look changed after a re-checkout, and once bytes are read there is nothing to gain from a
  timestamp.
- The boundary probe and memorisation baseline printed beside the headline number.
- Message-first form after the first end-to-end query showed the original six-field form was
  backwards for a real agent.
- Reading prose facts into blank structured fields.

**Rejected.**

- Hardcoding thresholds in the rule engine or the prompt. They are parsed from the documents.
- Publishing a comparison table with figures for a pipeline that does not exist. Every number
  here came from a run.
- Raising the evaluation bar to a 100% target on five cases; that adds overfitting pressure, not
  rigour.
- Growing the damage keyword list to include `spilled`. Applied once, reverted the same hour.
- Tuning the prompt so B09 passes.
- Claiming the prefix is cached, or recording a cached-token count.

**Bugs the checks caught that reading did not.**

1. `load_dotenv()` was reading a `.env` from outside the project.
2. `decision.py` used `config` without importing it - would have raised on the first real ticket.
3. The prompt listed the output keys but not the permitted actions.
4. The model path echoed the model's self-reported confidence, uncapped, at 1.0.
5. Every transport error fired a corrective retry, burning quota into a rate limit.
6. A rate-limited call was counted as a wrong answer: `12/16` when three cases had not run.
7. A subprocess test with a real key present silently made 21 live calls.
8. The front-matter bug described in section 6.
9. Defective tickets with an unknown value were approved without the required evidence.
10. The signal/cap followed the router's guess instead of the answered action (section 4).
11. `NEEDS_MORE_INFORMATION` was capped like a concrete action and reported 0.95 regardless of
    how little of the ticket was known (section 4).

---

## 8. Tests

One seam and one substitution point. Everything is asserted through the HTTP surface with
FastAPI's test client against a temporary database; the one substitution point is the model
transport in `src/llm.py`, faked in tests. The suite needs no API key and makes no network call.

Eight scenario tests: registration and sign-in, token rejection, tenant isolation, ticket
submission and persistence, the offline engine against the supplied cases and every boundary, the
policy documents and fingerprint, the model path (injection, retries, 503), and the required
evaluation report. Seven modules also carry assert-based self-checks:
`python -m src.config | src.cache | src.database | src.auth | src.models | src.decision |
src.retrieval`.
