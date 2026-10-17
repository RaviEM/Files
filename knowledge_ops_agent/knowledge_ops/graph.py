from langgraph.graph import END, START, StateGraph

from . import nodes
from .config import settings
from .state import AgentState

def after_validation(state: AgentState)-> str:
    return "memory" if state["input_ok"] else "finalize"

def after_verifier(state: AgentState)-> str:
    if state["verdict"]["passed"]:
        return "finalize"
    if state["retries"] <= settings.max_retries:
        return "orchestrator"
    return "finalize"

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("validate_input", nodes.validate_input)
    g.add_node("memory", nodes.memory_node)
    g.add_node("orchestrator", nodes.orchestrator)
    g.add_node("retriever", nodes.retriever)
    g.add_node("analyst", nodes.analyst)
    g.add_node("verifier", nodes.verifier)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "validate_input")
    g.add_conditional_edges("validate_input", after_validation, {"memory": "memory", "finalize": "finalize"})
    g.add_edge("memory", "orchestrator")
    g.add_edge("orchestrator", "retriever")
    g.add_edge("retriever", "analyst")
    g.add_edge("analyst", "verifier")
    g.add_conditional_edges("verifier", after_verifier, {"orchestrator": "orchestrator", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()