import json
import uuid

import streamlit as st

from knowledge_ops.config import settings

st.set_page_config(page_title="Knowledge Ops Agent", layout="wide")
st.title("Enterprise Knowledge Ops Agent")
st.caption(f"provider: {settings.llm_provider} - grounding threshold: "
            f"{settings.grounding_threshold} - max retries: {settings.max_retries}")

@st.cache_resource
def graph():
    from knowledge_ops.graph import build_graph
    return build_graph()

if "conv" not in st.session_state:
    st.session_state.conv = uuid.uuid4().hex[:8]
    st.session_state.runs = []

query= st.chat_input("Ask a question about the document corpus")
if query:
    with st.spinner("Running agents_"):
        state= graph().invoke({"query": query, "conversation_id": st.session_state.conv, "trace": [], "evaluation": []})
    st.session_state.runs.append(state)

for state in reversed (st.session_state.runs):
    left, right = st.columns([3, 2])
    with left:
        st.subheader(state["query"])
        badge = {"high": "High", "medium": "Medium", "low": "Low", "refused":"Refused"}
        st.markdown(f"{badge[state['confidence']]} **{state['confidence']}**")
        st.markdown(state["final_answer"])
    with right:
        with st.expander("Plan", expanded=False):
            st.json(state.get("plan") or {})
        with st.expander("Decision trace", expanded=True):
            for t in state["trace"]:
            st.markdown(f"`{t['step']:>2}` **{t['agent']}** - {t['action']} - {t['detail']}")
    with st.expander("Claim verification"):
        for c in (state.get("verdict") or {}).get("checks", []):
            st.markdown (f"- **{c['verdict']}** - {c['claim']}"
                        + (f" \n _evidence:_ {c['evidence']}" if c["evidence"] else ""))
    with st.expander("Evaluation signals"):
        st.json(state.get("evaluation", []))
    with st.expander ("Retrieved chunks");
        for ch in state.get("chunks", []):
            st.markdown (f"**{ch['chunk_id']}** score={ch['score']} ({ch['subtask_id']})")
            st.text(ch["text"][:400])
st.divider()