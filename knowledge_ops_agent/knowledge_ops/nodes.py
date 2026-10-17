import json

from . import guardrails, memory
from .config import settings
from .llm import structured
from .observability import evaluation, trace
from .prompts import (ANALYST_PROMPT, ORCHESTRATOR_PROMPT, REFINE_PROMPT, VERIFIER_PROMPT, Plan, Synthesis, Verdict)
from .state import AgentState

_vectorstore=None

def _store():
    global _vectorstore
    if _vectorstore is None:
        from .ingest import get_vectorstore
        _vectorstore= get_vectorstore()
    return _vectorstore

def _fmt_chunks(chunks) -> str:
    return "\n\n".join(f"[{c['chunk_id']}] ({c['source']}, p.{c['page']})\n{c['text']}"
                        for c in chunks)

def validate_input(state: AgentState)-> dict:
    ok, reason = guardrails.validate(state ["query"])
    return {
        "input_ok": ok, "rejection_reason": reason, "retries": 0,
        "retrieval_flags": [],
        "trace": trace(state, "guardrail", "validate_input",
                    "accepted" if ok else f"rejected: {reason}"),
    }

def memory_node(state: AgentState) -> dict:
    hist = memory.load_history(state["conversation_id"])
    return {"history": hist,
            "trace": trace (state, "memory", "load_history",
                                    f"{len(hist)} prior turn(s) loaded")}

def orchestrator(state: AgentState) -> dict:
    llm = structured("orchestrator", Plan)
    if state.get("retries", 0) == 0:
        prompt = ORCHESTRATOR_PROMPT.format(
            history=json.dumps(state.get("history", []), indent=1),
            query=state["query"])
    else:
        unsupported = [c["claim"] for c in state ["verdict"]["checks"]
                        if c["verdict"] in ("unsupported", "contradicted")]
        prompt = REFINE_PROMPT.format(
            query=state["query"], plan=json.dumps(state["plan"]),
            unsupported=unsupported, flags=state.get("retrieval_flags", []))
    plan: Plan =  llm.invoke(prompt)
    detail = " | ".join(f"{t.id}: {t.question}" for t in plan.subtasks)
    return {"plan": plan.model_dump(),
            "trace": trace (state, "orchestrator", "plan", detail)}

def retriever(state: AgentState) -> dict:
    store = _store()
    chunks, flags, evals = [], [], []
    seen = set()
    for task in state["plan"]["subtasks"]:
        hits = store.similarity_search_with_relevance_scores(
            task ["question"], k=settings.top_k)
        scores = [round (s, 3) for _, s in hits]
        if not hits or max(scores) < settings.min_relevance:
            flags.append(f"{task ['id']}: low relevance (max={max (scores or [0])})")
        for doc, score in hits:
            cid = doc.metadata ["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            chunks.append({
                "chunk_id": cid, "source": doc.metadata["source"],
                "doc_type": doc.metadata.get("doc_type", ""),
                "page": doc.metadata.get("page", 0), "text": doc.page_content,
                "score": round(score, 3), "subtask_id": task ["id"]})
        evals += evaluation("retriever", "relevance_scores", scores,
                            subtask=task["id"])
    return {"chunks": chunks, "retrieval_flags": flags, "evaluation": evals,
        "trace": trace (state, "retriever", "search",
                        f"{len(chunks)} unique chunks from "
                        f"{len({c['source'] for c in chunks})} source(s)"
                        +(f"; flags: {flags}" if flags else ""))}

def analyst(state: AgentState) -> dict:
    llm = structured ("analyst", Synthesis)
    plan = state["plan"]
    out: Synthesis = llm.invoke(ANALYST_PROMPT.format(
        query=state["query"],
        synthesis_instruction=plan["synthesis_instruction"],
        subtasks=[t["question"] for t in plan["subtasks"]],
        chunks=_fmt_chunks(state["chunks"])))
    return {"draft_answer": out.answer,
            "claims": [c.model_dump() for c in out.claims],
            "noted_conflicts": out.noted_conflicts,
            "trace": trace(state, "analyst", "synthesize",
                            f"{len(out.claims)} claims; "
                            f"{len(out.noted_conflicts)} conflict(s) noted")}

def verifier(state: AgentState) -> dict:
    llm = structured("verifier", Verdict)
    v: Verdict = llm.invoke(VERIFIER_PROMPT.format(
        threshold=settings.grounding_threshold,
        claims=json.dumps(state["claims"], indent=1),
        chunks=_fmt_chunks(state["chunks"])))
    weights = {"supported": 1.0, "partially_supported": 0.5}
    score = (sum(weights.get(c.verdict, 0) for c in v.checks) / len(v.checks)
            if v.checks else 0.0)
    contradicted = any(c.verdict == "contradicted" for c in v.checks)
    passed = score >= settings.grounding_threshold and not contradicted
    v.grounding_score, v.passed = round(score, 3), passed
    conflicts = list(set(v.conflicts + state.get("noted_conflicts", [])))
    evals = evaluation("verifier", "grounding_score", v.grounding_score,
                        attempt=state.get("retries", 0))
    if conflicts:
        evals += evaluation ("verifier", "conflicts", conflicts) 
    if not passed:
        evals += evaluation ("verifier", "failure", "low_grounding" 
                            if not contradicted else "contradiction")
    return {"verdict": {**v.model_dump(), "conflicts": conflicts},
            "retries": state.get("retries", 0) + (0 if passed else 1),
            "evaluation": evals,
            "trace": trace(state, "verifier", "check",
                            f"grounding={v.grounding_score} passed={passed}: {v. reason}")}

def finalize(state: AgentState) -> dict:
    if not state.get("input_ok", True):
        answer = f"I can't process that request. {state ['rejection_reason']}"
        conf, sources = "refused", []
    else:
        v = state["verdict"]
        score= v["grounding_score"]
        conf = ("high" if v["passed"] and score >= 0.9 else
                "medium" if v["passed"] else
                "low" if score >= 0.5 else "refused")
        by_id = {c["chunk_id"]: c for c in state ["chunks"]}
        used = {cid for cl in state["claims"] for cid in cl["chunk_ids"] if cid in by_id}
        sources = sorted({f"{by_id[c]['source']} (p.{by_id[c]['page']})" for c in used})
        answer = state["draft_answer"]
        if conf == "refused":
            answer = ("I couldn't find enough grounded information in the "
                        "documents to answer this reliably.\n\nPartial findings: "
                        + answer)
        if conf in ("Low", "refused"):
            weak = [c["claim"] for c in v["checks"]
                    if c["verdict"] != "supported"]
            answer += ("\n\n Confidence is low. These statements could not be "
                        "fully verified against the sources: " + "; ".join(weak))
        if v["conflicts"]:
            answer += "\n\nNote: sources disagree on:" + "; ".join(v["conflicts"])
        answer += "\n\nSources: " + ", ".join(sources)
    memory.append_turn(state["conversation_id"], state["query"], answer, conf)
    return {"final_answer": answer, "sources": sources, "confidence": conf,
            "trace": trace(state, "finalize", "respond", f"confidence={conf}")}