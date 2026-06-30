from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, \
    HumanMessagePromptTemplate

from app.bot import RAGState
from app.bot.llm import llm_model, llm_retry, str_parser
from app.core import logger

_MAX_ANS_ITERATION = 3


async def rewrite_answer_router(state: RAGState):
    if state["answer_relevance"] == "FULLY_SUPPORTED":
        return "check_answer_usefulness"

    if state["ans_iteration"] < _MAX_ANS_ITERATION:
        return "rewrite_answer"

    return "check_answer_usefulness"

async def rewrite_answer(state: RAGState):
    logger.info("Rewriting answer...")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            # v2 — fluent prose rewrite instead of raw quotes; handles no-match and near-correct edge cases
            "You are a strict grounding reviser. Rewrite the ANSWER to remove all unsupported claims "
            "while keeping it readable and natural.\n\n"
            "Rules:\n"
            "- Keep every claim that is explicitly stated in CONTEXT. Use the same wording where possible.\n"
            "- Remove or replace any word, phrase, or inference not present in CONTEXT.\n"
            "- Write in fluent prose (full sentences). Do NOT produce a bullet list of raw quotes.\n"
            "- Do NOT add new information, explain, or speculate.\n"
            "- Do NOT reference the context, document, or source in your output.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', 'based on the document'.\n\n"
            "Edge case guardrails:\n"
            "- If CONTEXT contains no information relevant to the QUESTION → output exactly: "
            "'I was unable to find specific information about this in the available documents.'\n"
            "- If only one unsupported word needs removal, rewrite that sentence minimally — do not rewrite the whole answer.\n"
            "- If CONTEXT contradicts part of the ANSWER, keep only the version supported by CONTEXT.\n\n"
            "Example:\n"
            "ANSWER: 'NexaAI offers a generous 30-day refund policy for all plans.'\n"
            "CONTEXT: 'Refunds are available within 30 days of purchase for all subscription plans.'\n"
            "REWRITE: 'NexaAI offers a 30-day refund policy for all subscription plans.'"
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: \n{question}\n\n"
            "Answer: \n{answer}\n\n"
            "Context: \n{context}\n\n"
        )
    ])
    chain = prompt | llm_model | str_parser
    response = await llm_retry(chain.ainvoke)({
        "question": state["question"],
        "answer": state["answer"],
        "context": state["context_str"]
    })
    return {
        "answer": response,
        "ans_iteration": state["ans_iteration"] + 1
    }
