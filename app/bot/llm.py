from google.api_core.exceptions import ResourceExhausted
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, \
    wait_exponential

from app.core import settings

llm_model = ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL)
str_parser = StrOutputParser()

llm_retry = retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
