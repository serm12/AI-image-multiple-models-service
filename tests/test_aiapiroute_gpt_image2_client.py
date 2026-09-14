import base64
from io import BytesIO
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from app.clients.aiapiroute_gpt_image2_client import AIApiRouteGPTImageClient
from app.core.config import APIConfig


class AIApiRouteGPTImageClientTests(unittest.TestCase):
    def setUp(self):
        self.client = AIApiRouteGPTImageClient(
            api_key="test-key",
            base_url="https://example.test",
            model=APIConfig.AIAPIROUTE_GPT_IMAGE2_MODEL,
        )

    def test_request_aspect_ratio_overrides_configured_fallback(self):
        with patch.object(
            APIConfig, "AIAPIROUTE_GPT_IMAGE2_REFERENCE_RATIO", "3:4"
        ):
            self.assertEqual(self.client._resolve_reference_ratio("4:3"), "4:3")

    def test_reference_image_is_fitted_to_request_aspect_ratio_without_cropping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = f"{temp_dir}/source.png"
            Image.new("RGB", (800, 1200), "green").save(image_path)

            data_url = self.client._to_request_data_url(image_path, "4:3")
            encoded = data_url.split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as result:
                self.assertEqual(result.size, (1600, 1200))

    def test_missing_request_ratio_uses_configured_fallback(self):
        with patch.object(
            APIConfig, "AIAPIROUTE_GPT_IMAGE2_REFERENCE_RATIO", "3:4"
        ):
            self.assertEqual(self.client._resolve_reference_ratio(None), "3:4")


if __name__ == "__main__":
    unittest.main()
