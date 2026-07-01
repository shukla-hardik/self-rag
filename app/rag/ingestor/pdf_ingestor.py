import tempfile

from langchain_community.document_loaders import PyPDFLoader

from app.core import s3
from app.rag.ingestor.abstract import BaseIngestor


class PdfIngestor(BaseIngestor):

    async def _load_documents(self):
        content = await s3.download_bytes(self.file_path)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(content)
            tmp.flush()
            documents = await PyPDFLoader(file_path=tmp.name).aload()
        return documents
