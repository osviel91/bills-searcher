import os

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings

CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "invoice-pdfs")


def client():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING debe configurarse")
    return BlobServiceClient.from_connection_string(connection_string)


def container_client(service=None):
    return (service or client()).get_container_client(CONTAINER)


def ensure_container(service=None):
    container = container_client(service)
    try:
        container.create_container()
    except ResourceExistsError:
        pass
    return container


def put(key: str, data: bytes, content_type="application/pdf"):
    container_client().upload_blob(key, data, overwrite=True,
                                   content_settings=ContentSettings(content_type=content_type))


def get(key: str):
    return container_client().get_blob_client(key).download_blob()


def delete(key: str):
    container_client().delete_blob(key)
