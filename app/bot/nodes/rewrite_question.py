from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from app.bot import RAGState
from app.bot.llm import llm_model, llm_retry

_MAX_QUE_ITERATION = 3


async def rewrite_question_router(state: RAGState):
    if state["is_ans_useful"]:
        return "stream_answer"
    if state["rewrite_tries"] < _MAX_QUE_ITERATION:
        return "rewrite_question"
    return "no_answer_found"


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        description="Rewritten query optimized for vector retrieval against internal company PDFs."
    )


async def rewrite_question(state: RAGState):
    response = await llm_retry(
        llm_model.with_structured_output(RewriteDecision).ainvoke
    )([
            SystemMessage(
                "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
                "Rules:\n"
                "- Keep it short (6–16 words).\n"
                "- Preserve key entities (e.g., NexaAI, plan names).\n"
                "- Add 2–5 high-signal keywords that likely appear in policy/pricing docs.\n"
                "- Remove filler words.\n"
                "- Do NOT answer the question.\n"
                "- Output JSON with key: retrieval_query\n\n"
                "Examples:\n"
                "Q: 'Do NexaAI plans include a free trial?'\n"
                "-> {{'retrieval_query': 'NexaAI free trial duration trial period plans'}}\n\n"
                "Q: 'What is NexaAI refund policy?'\n"
                "-> {{'retrieval_query': 'NexaAI refund policy cancellation refund timeline charges'}}"
            ),
            HumanMessage.from_template(
                f"QUESTION:\n{state["question"]}\n\n"
                f"Previous retrieval query:\n{state.get("retrieval_query")}\n\n"
                f"Answer (if any):\n{state["answer"]}"
            )
        ])
    return {
        "retrieval_query": response.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context_str": ""
    }
