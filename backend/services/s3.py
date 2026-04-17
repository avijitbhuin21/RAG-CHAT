import boto3
from botocore.config import Config

from ..config import settings

_endpoint = settings.S3_ENDPOINT
if not _endpoint.startswith("http"):
    _endpoint = ("https://" if settings.S3_SECURE else "http://") + _endpoint

_client = boto3.client(
    "s3",
    endpoint_url=_endpoint,
    aws_access_key_id=settings.S3_ACCESS_KEY,
    aws_secret_access_key=settings.S3_SECRET_KEY,
    region_name=settings.S3_REGION,
    config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
)


def head_bucket() -> None:
    _client.head_bucket(Bucket=settings.S3_BUCKET)


def delete_object(key: str) -> None:
    """boto3's delete_object is idempotent — missing key returns silently."""
    _client.delete_object(Bucket=settings.S3_BUCKET, Key=key)


def client():
    return _client
