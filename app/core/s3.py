from app.core.aws import s3_client
from app.core.config import settings


async def upload_bytes(key: str, content: bytes) -> None:
    async with s3_client() as s3:
        await s3.put_object(Bucket=settings.S3_BUCKET_NAME, Key=key, Body=content)


async def download_bytes(key: str) -> bytes:
    async with s3_client() as s3:
        response = await s3.get_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
        async with response["Body"] as stream:
            return await stream.read()


async def delete_object(key: str) -> None:
    async with s3_client() as s3:
        await s3.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=key)