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

    def test_native_aspect_ratio_and_resolution_are_sent_without_derived_size(self):
        self.assertEqual(
            self.client._resolve_request_options("2K", "4:3"),
            {"aspect_ratio": "4:3", "resolution": "2K"},
        )

    def test_explicit_pixel_size_remains_supported(self):
        self.assertEqual(
            self.client._resolve_request_options("1536x1024", "3:2"),
            {"aspect_ratio": "3:2", "size": "1536x1024"},
        )

    def test_reference_image_compatibility_fit_remains_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = f"{temp_dir}/source.png"
            Image.new("RGB", (800, 1200), "green").save(image_path)

            data_url = self.client._prepare_reference_image(image_path, "4:3")
            encoded = data_url.split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as result:
                self.assertEqual(result.size, (1600, 1200))

    def test_gpt_image_25_providers_are_registered(self):
        flare_provider = "gpt-image-2.5-flare_aiapiroute"
        sunburst_provider = "gpt-image-2.5-sunburst_aiapiroute"

        self.assertIn(flare_provider, APIConfig.ALL_PROVIDERS)
        self.assertIn(sunburst_provider, APIConfig.ALL_PROVIDERS)
        self.assertEqual(
            APIConfig.AIAPIROUTE_PROVIDER_MODEL_MAP[flare_provider],
            APIConfig.AIAPIROUTE_GPT_IMAGE25_FLARE_MODEL,
        )
        self.assertEqual(
            APIConfig.AIAPIROUTE_PROVIDER_MODEL_MAP[sunburst_provider],
            APIConfig.AIAPIROUTE_GPT_IMAGE25_SUNBURST_MODEL,
        )
        self.assertEqual(
            APIConfig.resolve_provider(flare_provider, validate_config=False),
            flare_provider,
        )
        self.assertEqual(
            APIConfig.resolve_provider(sunburst_provider, validate_config=False),
            sunburst_provider,
        )
        self.assertEqual(
            APIConfig.resolve_provider("gpt-image-2.5", validate_config=False),
            flare_provider,
        )


if __name__ == "__main__":
    unittest.main()
