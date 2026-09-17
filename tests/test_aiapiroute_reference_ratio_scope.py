import unittest
from unittest.mock import patch

from app.clients.aiapiroute_gpt_image2_client import AIApiRouteGPTImageClient
from app.core.config import APIConfig


class AIApiRouteReferenceRatioScopeTests(unittest.TestCase):
    def test_reference_ratio_compatibility_flag_applies_to_aiapiroute_models(self):
        client = AIApiRouteGPTImageClient(
            api_key="test-key",
            base_url="https://example.test",
            model="gpt-image-2.5",
        )

        with patch.object(APIConfig, "AIAPIROUTE_GPT_IMAGE2_FORCE_REFERENCE_RATIO", True):
            self.assertTrue(client._should_force_reference_ratio())


if __name__ == "__main__":
    unittest.main()
