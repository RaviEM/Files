# Enterprise Knowledge Ops Agent
Local multi-agent RAG system built with LangGraph. 
Five agents (orchestrator, retriever, analyst, verifier, memory) plus a guardrail gate answer questions across enterprise documents with source attribution, grounding checks and a full decision trace.

## Quick start (fully local)
```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b
cp .env.example .env
python main.py ingest
# or set LLM_PROVIDER-azure in .env
# indexes ./data into ./chroma_db
python main.py ask "Who approves a USD 120,000 vendor contract and what checks are needed first?"
python -m pytest tests
```
## Layout
| Path | Purpose |
|---|---|
| `knowledge_ops/graph.py` | Node wiring and routing (orchestration decisions) |
| `knowledge_ops/nodes.py` | Agent implementations |
| `knowledge_ops/prompts.py` | Prompts and Pydantic output schemas |
| `knowledge_ops/state.py` | Shared AgentState' |
| `knowledge_ops/ingest.py` | Load chunk embed Chroma |
| `knowledge_ops/guardrails.py` | Input validation |
| `knowledge_ops/observability.py` | Trace / evaluation log writer |
| `knowledge_ops/memory.py` | Per-conversation history |
| `app.py` | Streamlit debug UI (streamlit run app.py`) |
| `scripts/evaluate.py` | Golden-set evaluation ('python -m scripts.evaluate`) |
| `docs/` | Architecture, evaluation & guardrails, testing docs | 
| `logs/run_*.json` | One structured record per query |

## Flow
validate input -> memory -> orchestrator -> retriever -> analyst -> verifier -> finalize

If the verifier fails (grounding below 'GROUNDING THRESHOLD or a contradicted claim) it routes back to the orchestrator, which re-plans around the unsupported claims, up to 'MAX_RETRIES' times. After that, finalize answers with low confidence and lists the unverified statements.

## Evaluation signals (all in `logs/run_*.json`)

- `retriever.relevance_scores` per sub-task, with low-relevance flags
- `verifier.grounding_score` per attempt, recomputed deterministically
- `verifier.conflicts` when sources disagree
- `verifier.failure` = `low_grounding` or `contradiction`
- `trace`  = ordered agent decisions