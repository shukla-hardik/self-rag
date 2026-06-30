import asyncio

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, \
    HumanMessagePromptTemplate
from pydantic import BaseModel, Field

from app.bot import RAGState
from app.bot.llm import llm_model, llm_retry
from app.core import logger


class RelevanceCheckerModel(BaseModel):
    is_relevant: bool = Field(
        description="True, if the given document is relevant to answer the "
                    "question, else False"
    )


async def context_relevance_checker(state: RAGState):
    logger.info("Checking doc relevance...")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            # v2 — stricter threshold, few-shot examples, edge case guardrails
            "You are judging whether a document contains the specific information needed to answer a question.\n\n"
            "is_relevant=true ONLY if the document directly contains facts, figures, policy details, or procedures "
            "that would appear in a correct answer.\n"
            "is_relevant=false if the document only mentions the topic in passing, provides background without "
            "the specific fact, or is about a related but different subject.\n\n"
            "Edge cases:\n"
            "- Document is empty or fewer than 10 words → false\n"
            "- Document mentions the entity (e.g. 'NexaAI') but not the specific fact asked → false\n"
            "- Document partially covers the question (answers one part, silent on another) → true\n"
            "- Question is ambiguous but document covers the most likely interpretation → true\n\n"
            "Examples:\n"
            "Q: 'What is the refund policy?'\n"
            "Doc: 'NexaAI offers flexible pricing plans for teams of all sizes.' → false\n\n"
            "Q: 'What is the refund policy?'\n"
            "Doc: 'Refunds are processed within 7 business days of a cancellation request.' → true\n\n"
            "Q: 'Does NexaAI support SSO?'\n"
            "Doc: 'NexaAI Enterprise plan includes SAML-based single sign-on integration.' → true\n\n"
            "Q: 'How many users per plan?'\n"
            "Doc: 'NexaAI was founded in 2019 and serves over 500 companies.' → false"
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: \n{question}\n\n"
            "Document: \n{document}"
        )
    ])
    llm = llm_model.with_structured_output(RelevanceCheckerModel)
    chain = prompt | llm

    coroutines = []
    for doc in state["docs"]:
        coroutines.append(llm_retry(chain.ainvoke)({
            "question": state["question"],
            "document": doc.page_content
        }))

    decisions = await asyncio.gather(*coroutines)

    relevant_docs = []
    for doc, decision in zip(state["docs"], decisions):
        if decision.is_relevant:
            relevant_docs.append(doc)

    logger.info(f"Relevant docs found: {len(relevant_docs)}/"
                f"{len(state["docs"])}")
    return {
        "relevant_docs": relevant_docs,
        "context_str": "\n\n".join(map(
            lambda x: x.page_content,
            relevant_docs
        )).strip()
    }
