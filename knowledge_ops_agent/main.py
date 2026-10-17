import argparse
import json
import uuid

from knowledge_ops.config import settings

def run_query(graph, query: str, conv: str, show_trace: bool = True)-> dict:
    from knowledge_ops.observability import write_run_log
    state = graph.invoke({"query": query, "conversation_id": conv,
                            "trace": [], "evaluation": []})
    print("\n" + state ["final_answer"])
    print(f"\n [confidence: {state['confidence']}]")
    if show_trace:
        print("\nDecision trace:")
        for t in state["trace"]:
            print(f" {t['step']:>2}. {t['agent']:<12} {t['action']: <14} {t['detail']}")
    path = write_run_log(state)
    print(f"\nRun log: {path}")
    return state



def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    a = sub.add_parser("ask"); a.add_argument ("query"); a.add_argument("--conv")
    a.add_argument("--no-trace", action="store_true")
    c = sub.add_parser("chat"); c.add_argument("--conv")
    args = p.parse_args()

    if args.cmd == "ingest":
        from knowledge_ops.ingest import ingest
        print(f"Ingested {ingest()} chunks into {settings.chroma_dir}")
        return
    
    from knowledge_ops.graph import build_graph
    graph = build_graph()
    conv = args.conv or uuid.uuid4().hex[:8]
    print(f"conversation: {conv} provider: {settings.llm_provider}")

    if args.cmd="ask":
        run_query(graph, args.query, conv, not args.no_trace)
    else:
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", ""):
                break
            run_query(graph, q, conv)



if __name__ == "__main__":
    main()