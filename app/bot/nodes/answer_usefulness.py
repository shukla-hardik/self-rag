from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from app.bot import RAGState
from app.bot.llm import llm_model, llm_retry
from app.core import logger


class AnswerUsefulModel(BaseModel):
    is_ans_useful: bool
    reason: str


async def check_answer_usefulness(state: RAGState):
    logger.info("Checking for the answer's usefulness...")
    response: AnswerUsefulModel = await llm_retry(
        llm_model.with_structured_output(AnswerUsefulModel).ainvoke
    )([
        SystemMessage(
            # v2 — fallback string guardrails, concrete examples for borderline cases
            "You are judging whether the ANSWER actually addresses what the user asked in QUESTION.\n\n"
            "is_ans_useful=true: ANSWER directly responds to the question with specific information.\n"
            "is_ans_useful=false: ANSWER is generic, off-topic, only gives background, or admits it cannot answer.\n\n"
            "Edge case guardrails:\n"
            "- ANSWER is 'No relevant document found' or 'I don't know' → false\n"
            "- ANSWER is empty or blank → false\n"
            "- ANSWER gives a related fact but not the one asked (e.g. asked about refunds, answered about cancellation) → false\n"
            "- ANSWER partially addresses the question (e.g. answers yes/no but omits the requested detail) → false\n"
            "- ANSWER is technically correct but for a subtly different question → false\n\n"
            "Examples:\n"
            "Q: 'Does NexaAI offer a free trial?'\n"
            "A: 'NexaAI provides various options for new customers.' → false (generic, no direct answer)\n\n"
            "Q: 'Does NexaAI offer a free trial?'\n"
            "A: 'Yes, NexaAI offers a 14-day free trial on all plans.' → true\n\n"
            "Q: 'What is the refund timeline?'\n"
            "A: 'NexaAI has a refund policy for cancelled subscriptions.' → false (mentions topic, no specific answer)\n\n"
            "Rules:\n"
            "- DO NOT re-check grounding. Only check: did we answer the question?\n"
            "- DO NOT use outside knowledge.\n"
            "- Keep reason to 1 line."
        ),
        HumanMessage(
            f"Question: {state["question"]}\n\n"
            f"Answer: {state["answer"]}"
        )
    ])
    return {
        "is_ans_useful": response.is_ans_useful,
        "reason": response.reason
    }
