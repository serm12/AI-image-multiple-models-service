import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.clients.seedream5pro_fal_client import Seedream5ProFalClient
from app.core.config import APIConfig


class Seedream5ProFalClientTests(unittest.TestCase):
    def test_uses_the_requested_fal_endpoint_by_default(self):
        with patch.object(APIConfig, "FAL_API_KEY", "test-key"):
            client = Seedream5ProFalClient()
        self.assertEqual(client.model_id, "bytedance/seedream/v5/pro/edit")

    def test_uses_portrait_size_for_three_by_four(self):
        self.assertEqual(
            Seedream5ProFalClient.image_size_for_ratio("3:4"),
            {"width": 1536, "height": 2048},
        )

    def test_match_input_has_a_supported_portrait_fallback(self):
        self.assertEqual(
            Seedream5ProFalClient.image_size_for_ratio("match_input_image"),
            {"width": 1536, "height": 2048},
        )

    def test_provider_is_registered_and_resolves(self):
        provider = "seedream-5-pro_fal"
        self.assertIn(provider, APIConfig.ALL_PROVIDERS)
        self.assertEqual(APIConfig.resolve_provider(provider, validate_config=False), provider)
        self.assertEqual(APIConfig.resolve_provider("seedream-5-pro", validate_config=False), provider)

    def test_submits_the_pro_endpoint_with_compatible_edit_input(self):
        handler = Mock()
        handler.get.return_value = {"images": [{"url": "https://example.test/image.png"}]}
        with patch.object(APIConfig, "FAL_API_KEY", "test-key"):
            client = Seedream5ProFalClient()
        with patch.object(client, "_upload_local_image", AsyncMock(return_value="https://example.test/source.jpg")):
            with patch("app.clients.seedream5pro_fal_client.fal_client.submit", return_value=handler) as submit:
                result = asyncio.run(client.generate_image(
                    prompt="paint the supplied portrait",
                    input_image_paths=["portrait.jpg"],
                    aspect_ratio="3:4",
                ))

        submit.assert_called_once_with("bytedance/seedream/v5/pro/edit", {
            "prompt": "paint the supplied portrait",
            "image_urls": ["https://example.test/source.jpg"],
            "image_size": {"width": 1536, "height": 2048},
            "num_images": 1,
            "output_format": "png",
            "enable_safety_checker": True,
        })
        self.assertEqual(result["output"], "https://example.test/image.png")


if __name__ == "__main__":
    unittest.main()
