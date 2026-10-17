from typing import Literal

from pydantic import BaseModel, Field

class PlannedSubTask(BaseModel):
    id: str = Field(description="Short id, e.g. 't1'")
    question: str = Field(description="Self-contained sub-question")
    rationale: str = Field(description="Why this step is needed (1 sentence)")
    depends_on: list[str] = Field(default_factory=list,
                                description="ids that must finish first")

class Plan(BaseModel):
    intent: str = Field(description="One-line restatement of what the user wants")
    needs_multiple_documents: bool
    subtasks: list[PlannedSubTask] = Field(min_length=1, max_length=5)
    synthesis_instruction: str = Field(
        description="How the analyst should combine sub-answers")

ORCHESTRATOR_PROMPT="""You are the planning agent in an enterprise knowledge system.
The corpus contains policies, SOPs, contracts, compliance manuals and audit reports.
Your job is to decompose the user's question into the smallest set of retrievable
sub-questions (1 to 5) such that answering all of them fully answers the original.

Rules:
- Each sub-question must be answerable by searching documents; do not plan steps that require outside knowledge or calculation the documents cannot support.
- Split only where different documents or sections are likely needed. A simple factual question is ONE subtask.
- Mark depends_on when a later step needs the result of an earlier one (e.g. "find the policy owner" before "find that owner's approval limit").
- Prefer concrete nouns from the question in each sub-question so retrieval matches document wording.
- Write a synthesis_instruction telling the analyst how to combine results (compare, reconcile conflicts, list conditions, etc.).

Conversation history (may be empty):
{history}

User question:
{query}
"""


REFINE_PROMPT="""The previous attempt produced claims that could not be grounded.

Original question: {query}
Previous plan: {plan}
Unsupported claims: {unsupported}
Retrieval flags: {flags}

Produce a revised plan whose sub-questions specifically target the unsupported claims.
Reword to match likely document language. Do not repeat sub-questions that already retrieved well.
"""


class Claim(BaseModel):
    text: str = Field(description="One atomic factual statement")
    chunk_ids: list[str] = Field(description="Chunks this claim is based on")


class Synthesis(BaseModel):
    answer: str = Field(description="Full answer in prose, no citations yet")
    claims: list[Claim]
    noted_conflicts: list[str] = Field(
        default_factory=list,
        description="Places where sources disagree, stated plainly")


class ClaimCheck (BaseModel):
    claim: str
    verdict: Literal["supported", "partially_supported", "unsupported", "contradicted"]
    evidence: str = Field(description="Quote or paraphrase from the chunk, or ''")
    chunk_id: str | None


class Verdict(BaseModel):
    checks: list[ClaimCheck]
    grounding_score: float = Field(ge=0, le=1, description="supported / total claims")
    conflicts: list[str]
    passed: bool
    reason: str = Field(description="One sentence a reviewer can read")

VERIFIER_PROMPT="""You are a strict fact-checker. You do NOT answer the question.
You decide whether each claim is supported by the retrieved text and nothing else.


For every claim:
- supported: the cited chunk states it or it follows directly.
- partially_supported: part is stated, part is inferred or missing.
- unsupported: nothing in the cited chunks backs it.
- contradicted: the chunks say the opposite.

Then compute grounding_score = supported / total (partials count 0.5). Set passed = true only if grounding_score >= {threshold} and no claim is contradicted. List conflicts where two chunks disagree with each other.

Claims:
{claims}

Retrieved chunks (id, source, text):
{chunks}
"""

ANALYST_PROMPT = """You are the analyst agent. Answer the user's question using ONLY the retrieved chunks below. Follow the synthesis instruction from the planner.

Rules:
- Combine information across chunks; do not just copy one passage.
- Break your answer into atomic claims and list, for each, the chunk ids it rests on. Every sentence of the answer should correspond to a claim.
- If chunks disagree, say so in noted_conflicts and present both positions.
- If the chunks do not contain enough to answer, say exactly what is missing; never fill gaps from general knowledge.

Question: {query}
Synthesis instruction: {synthesis_instruction}
Sub-questions planned: {subtasks}
Retrieved chunks (id | source | text):
{chunks}
"""