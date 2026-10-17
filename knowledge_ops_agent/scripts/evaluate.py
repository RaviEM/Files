import json
import sys
from pathlib import Path

from knowledge_ops.graph import build_graph

path = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/golden.jsonl")
graph = build_graph()
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

scores, src_hits, conf_hits, retries = [], 0, 0, []
for r in rows:
    s = graph.invoke({"query": r["question"], "conversation_id": "eval", "trace": [], "evaluation": []})
    v = s.get("verdict") or {}
    scores.append(v.get("grounding_score", 0.0))
    cited = {x.split(" (p.")[0] for x in s.get("sources", [])}
    src_ok = set(r["expected_sources"]).issubset(cited)
    conf_ok = s["confidence"] in r["expected_confidence"]
    src_hits += src_ok; conf_hits += conf_ok; retries.append(s.get("retries", 0))
    print(f"{'OK' if src_ok and conf_ok else 'MISS'} {s['confidence']:<7}"
            f"g={scores[-1]:.2f} r={retries[-1]} {r['question'][:60]}")

n = len(rows)
print(f"\nmean grounding         {sum(scores)/n:.3f}")
print(f"expected sources hit    {src_hits}/{n}") 
print(f"confidence matched       {conf_hits}/{n}")
print(f"mean retries             {sum(retries)/n:.2f}")