from uuid import UUID

from fastapi import HTTPException, UploadFile
from starlette import status

from app.core import logger
from app.core import s3
from app.db.services import DocumentService
from app.worker import sqs_queue


class DocumentController:
    def __init__(self, db):
        self.db = db
        self.document_service = DocumentService(db)

    async def get_document(self, user_id: UUID, document_id: UUID):
        return await self.document_service.get_by_filter(
            id=document_id,
            user_id=user_id
        )

    async def list_all_documents(self, user_id: UUID):
        document_list = await self.document_service.get_all_by_filter(
            user_id=user_id)
        return document_list

    async def delete_document(self, user_id: UUID, document_id: UUID):
        document = await self.document_service.get_by_filter(
            id=document_id,
            user_id=user_id
        )
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        await self.document_service.delete_by_id(id=document_id)
        await s3.delete_object(document.file_path)
        logger.info(f"Deleted S3 object: {document.file_path}")

        logger.info(f"Deleted : {document_id}")

    async def handle_document_ingestion(
            self,
            file: UploadFile,
            user_id: UUID,
            doc_type: str | None = None,
            source: str | None = None,
    ):
        filename: str = file.filename
        s3_key: str = f"{user_id}/{filename}"

        if await self.document_service.is_ingested(
                user_id=user_id,
                filename=filename
        ):
            raise Exception("Given file is already processed")

        # Currently only supported for pdf
        await self._validate_pdf_doc(file)

        MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds the 2 MB limit"
            )

        # Uploading file to S3 (localstack in local dev)
        await s3.upload_bytes(s3_key, content)

        # Updating DB with the records
        doc_data = {
            "user_id": user_id,
            "filename": filename,
            "file_path": s3_key,
        }
        if doc_type:
            doc_data["doc_type"] = doc_type
        if source:
            doc_data["source"] = source
        document = await self.document_service.create(doc_data)

        await sqs_queue.send_ingest_message(
            document_id=document.id,
            user_id=user_id,
            filename=filename,
            s3_key=s3_key,
        )

    async def _validate_pdf_doc(self, file):
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files allowed"
            )

        content = await file.read(4)
        if not content.startswith(b"%PDF"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file"
            )

        await file.seek(0)
