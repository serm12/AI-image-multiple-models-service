import os
import unittest
from unittest.mock import patch

from app.core.config import env_float


class WatermarkConfigTests(unittest.TestCase):
    def test_center_logo_scale_reads_valid_value(self):
        with patch.dict(os.environ, {"WATERMARK_CENTER_LOGO_SCALE": "0.5"}):
            value = env_float("WATERMARK_CENTER_LOGO_SCALE", 1.0, 0.05, 1.0)
        self.assertEqual(value, 0.5)

    def test_scale_is_bounded_and_invalid_value_uses_default(self):
        with patch.dict(os.environ, {"WATERMARK_CENTER_LOGO_SCALE": "2"}):
            bounded = env_float("WATERMARK_CENTER_LOGO_SCALE", 1.0, 0.05, 1.0)
        with patch.dict(os.environ, {"WATERMARK_CENTER_LOGO_SCALE": "invalid"}):
            fallback = env_float("WATERMARK_CENTER_LOGO_SCALE", 0.75, 0.05, 1.0)

        self.assertEqual(bounded, 1.0)
        self.assertEqual(fallback, 0.75)

    def test_watermark_output_scale_reads_valid_value(self):
        with patch.dict(os.environ, {"WATERMARK_OUTPUT_SCALE": "0.75"}):
            value = env_float("WATERMARK_OUTPUT_SCALE", 0.5, 0.1, 1.0)
        self.assertEqual(value, 0.75)


if __name__ == "__main__":
    unittest.main()
