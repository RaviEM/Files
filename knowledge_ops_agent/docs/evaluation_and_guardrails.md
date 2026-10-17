# Evaluation, guardrails and observability 

## 1. Guardrails
### Input validation (`guardrails.py', runs before any LLM call)

| Check | Rule | Action |
|---|---|---|
Empty / oversize | `len(query) > MAX_QUERY_CHARS` (2000) | refuse |
| Prompt injection | regex list: "ignore previous instructions", "system prompt", "developer mode", - | refuse, reason Logged I
| Sensitive identifiers | SSN-like and card-like number patterns | refuse |

Refusals route directly to `finalize` with `confidence = refused`; no agent is invoked and the rejection reason is written to the trace.

### Output controls (`verifier` and `finalize`)

| Control | Mechanism |
|---|---|
| Grounding gate | answer is released only if `grounding_score > GROUNDING THRESHOLD` (0.8) and no clain is `contradicted` |
| Deterministic scoring | the LLM labels each claim; the score and pass/fail are recomputed in Python (`supported = 1`, `partially_supported = 0.5`, else `0`) |
| Bounded retry | failed verdicts re-plan at most `MAX_RETRIES` (2) times |
| Confidence Labelling | `high >= 0.9` - `medium` passed - `low >= 0.5` - `refused < 0.5` |
| Disclaimers | Low/refused answers list the unverified statements verbatim |
| Conflict disclosure | source disagreements (from analyst or verifier) are appended to the answer |
| Source attribution | sources are derived from the chunk ids the claims cite, not from everything retrieved 1
| Closed-world prompting | analyst prompt forbids filling gaps from general knowledge |

## 2. Evaluation signals
Every run writes `logs/run_<timestamp>.json`. The `evaluation` list contains:

| Signal | Emitted by | Shape | Detects |
|---|---|---|---|
| `relevance_scores` | retriever | per sub-task list of similarity scores | insufficient / irrelevant retrieval (flag when `max < MIN_RELEVANCE`) |
| `grounding_score` | verifier | float per attempt | low grounding |
| `conflicts` | verifier | list of strings contradictory sources or analyst/verifier disagreement |
| `failure` | verifier | `low_grounding` or `contradiction` | failed attempt |


## 3. Running an evaluation
A small golden set lives in `tests/golden.jsonl` (question, expected sources, expected confidence). `python -m scripts.evaluate` runs each question through the graph and reports:

- mean grounding score
- of answers that cited every expected source
- confidence labels matching expectation
- mean retries

Use it to compare models (`OLLAMA_MODEL`) or thresholds (`GROUNDING_THRESHOLD`) across runs.

## 4. Known limitations and roadmap
- Injection detection is regex-based; a classifier would catch paraphrased attacks.
- No automatic retry on structured-output parse failure; small local models occasionally need one.
- Grounding is judged by an LLM; an NLI model could pre-filter cheaply and reduce judge calls.
- Memory is a flat JSON history; no summarisation of long conversations yet.