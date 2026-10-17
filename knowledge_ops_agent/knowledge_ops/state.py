import operator
from typing import Annotated, Literal, TypedDict

class SubTask(TypedDict):
    id: str
    question: str
    rationale: str
    depends_on: list[str]


class RetrievedChunk(TypedDict):
    chunk_id: str
    source: str
    doc_type: str
    page: int
    text: str
    score: float
    subtask_id: str


class TraceEvent(TypedDict):
    step: int
    agent: str
    action: str
    detail: str


class AgentState(TypedDict, total=False):
    query: str
    conversation_id: str

    input_ok: bool
    rejection_reason: str | None

    history: list[dict]
    plan: dict
    retries: int
    retrieval_flags: list[str]

    chunks: list[RetrievedChunk]
    draft_answer: str
    claims: list (dict)
    noted_conflicts: list[str]

    verdict: dict | None

    trace: Annotated[list[TraceEvent], operator.add]
    evaluation: Annotated[list [dict], operator.add]


    final_answer: str
    sources: list[str]
    confidence: Literal["high", "medium", "low", "refused"]