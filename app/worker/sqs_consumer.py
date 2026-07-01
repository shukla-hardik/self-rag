import asyncio
import json
import signal

from app.core import logger
from app.db import DBClient
from app.rag import PdfIngestor
from app.worker import sqs_queue

_shutdown = asyncio.Event()


def _request_shutdown(*_args):
    _shutdown.set()


async def _process_message(message: dict) -> None:
    body = json.loads(message["Body"])
    document_id = body["document_id"]

    async with DBClient.get_session() as db:
        try:
            logger.info(f"Processing {document_id}")
            ingestor = PdfIngestor(
                db=db,
                user_id=body["user_id"],
                document_id=document_id,
                filename=body["filename"],
                file_path=body["s3_key"],
            )
            await ingestor.ainvoke()
            await db.commit()
        except Exception as exc:
            await db.rollback()
            logger.error(f"Failed to process document {document_id}: {exc}")
            # Leave the message in the queue: it becomes visible again after the
            # visibility timeout and SQS's redrive policy moves it to the DLQ
            # after the configured max receive count.
            return

    await sqs_queue.delete_message(message["ReceiptHandle"])


async def run() -> None:
    logger.info("SQS consumer started, polling for ingest messages")

    while not _shutdown.is_set():
        try:
            messages = await sqs_queue.receive_messages()
        except Exception as exc:
            logger.error(f"Failed to poll SQS: {exc}")
            await asyncio.sleep(5)
            continue

        for message in messages:
            await _process_message(message)

    logger.info("SQS consumer shutting down")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)
    loop.run_until_complete(run())
