from collections.abc import Iterator

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from kadastra.adapters.s3_raw_data import S3RawData

BUCKET = "test-bucket"


@pytest.fixture(autouse=True)
def s3_with_objects() -> Iterator[None]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        client.put_object(Bucket=BUCKET, Key="data/file1.csv", Body=b"col1,col2\na,1\n")
        client.put_object(Bucket=BUCKET, Key="data/file2.json", Body=b'{"k": "v"}')
        client.put_object(Bucket=BUCKET, Key="other/file3.txt", Body=b"hello")
        yield


def _adapter() -> S3RawData:
    return S3RawData(
        bucket=BUCKET,
        access_key="testing",
        secret_key="testing",
        region="us-east-1",
        addressing_style="path",
    )


def test_read_bytes_returns_object_content() -> None:
    adapter = _adapter()

    content = adapter.read_bytes("data/file2.json")

    assert content == b'{"k": "v"}'


def test_list_keys_returns_objects_under_prefix() -> None:
    adapter = _adapter()

    keys = adapter.list_keys("data/")

    assert sorted(keys) == ["data/file1.csv", "data/file2.json"]


def test_list_keys_does_not_return_objects_outside_prefix() -> None:
    adapter = _adapter()

    keys = adapter.list_keys("data/")

    assert "other/file3.txt" not in keys


def test_list_keys_returns_empty_list_for_unknown_prefix() -> None:
    adapter = _adapter()

    keys = adapter.list_keys("nonexistent/")

    assert keys == []


def test_read_bytes_raises_client_error_for_missing_key() -> None:
    adapter = _adapter()

    with pytest.raises(ClientError):
        adapter.read_bytes("missing.txt")
