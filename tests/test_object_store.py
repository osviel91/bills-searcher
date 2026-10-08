import unittest
from unittest.mock import Mock, patch

from azure.core.exceptions import ResourceExistsError

import object_store


class ObjectStoreTests(unittest.TestCase):
    def test_container_creation_is_safe_when_app_and_worker_start_together(self):
        service = Mock()
        container = service.get_container_client.return_value

        with patch.object(object_store, "CONTAINER", "test-pdfs"):
            self.assertIs(object_store.ensure_container(service), container)
            container.create_container.assert_called_once_with()

            container.create_container.side_effect = ResourceExistsError("already created")
            self.assertIs(object_store.ensure_container(service), container)
            self.assertEqual(container.create_container.call_count, 2)

    def test_upload_download_and_delete_use_the_configured_container(self):
        container = Mock()
        with patch.object(object_store, "CONTAINER", "test-pdfs"), \
             patch.object(object_store, "container_client", return_value=container):
            object_store.put("documents/one.pdf", b"%PDF")
            object_store.get("documents/one.pdf")
            object_store.delete("documents/one.pdf")

        container.upload_blob.assert_called_once()
        self.assertEqual(container.upload_blob.call_args.args[:2], ("documents/one.pdf", b"%PDF"))
        self.assertEqual(container.upload_blob.call_args.kwargs["content_settings"].content_type, "application/pdf")
        container.get_blob_client.assert_called_once_with("documents/one.pdf")
        container.get_blob_client.return_value.download_blob.assert_called_once_with()
        container.delete_blob.assert_called_once_with("documents/one.pdf")


if __name__ == "__main__":
    unittest.main()
