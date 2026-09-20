import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.core.config import WatermarkConfig, env_float
from PIL import Image, ImageChops, ImageFont
from app.utils.watermark_utils import (
    _scale_alpha,
    _uniform_tile_positions,
    add_corner_label,
    temp_add_corner_label,
)


class WatermarkConfigTests(unittest.TestCase):
    def test_tiled_watermark_balances_visibility_and_spacing(self):
        self.assertEqual(WatermarkConfig.DEFAULT_COLOR, (255, 255, 255, 153))
        self.assertEqual(WatermarkConfig.DEFAULT_STEP, 165)

    def test_corner_label_keeps_proportions_on_high_resolution_images(self):
        with tempfile.TemporaryDirectory() as directory:
            bounds = []
            for scale in (1, 2, 3):
                source = str(Path(directory) / 'source.png')
                output = str(Path(directory) / 'label.png')
                Image.new('RGB', (1024 * scale, 1536 * scale)).save(source)
                add_corner_label(source, output)
                with Image.open(output) as rendered:
                    rgb = rendered.convert('RGB')
                    box = ImageChops.difference(rgb, Image.new('RGB', rgb.size)).getbbox()
                    self.assertLess(box[2], rgb.width)
                    bounds.append(tuple(value / scale for value in box))
            for box in bounds[1:]:
                for actual, expected in zip(box, bounds[0]):
                    # Font hinting can slightly change glyph widths at each size.
                    self.assertAlmostEqual(actual, expected, delta=max(2, expected * 0.02))

    def test_legacy_corner_label_and_missing_font_use_same_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / 'source.png')
            output = str(Path(directory) / 'label.png')
            legacy = str(Path(directory) / 'legacy.png')
            Image.new('RGB', (1024, 1536)).save(source)
            add_corner_label(source, output)
            temp_add_corner_label(source, legacy, font_path='missing-font.ttf')
            with Image.open(output) as current, Image.open(legacy) as fallback:
                self.assertIsNone(ImageChops.difference(current.convert('RGB'), fallback.convert('RGB')).getbbox())

    def test_enlarged_corner_label_fits_portrait_width(self):
        font = ImageFont.truetype(
            WatermarkConfig.CORNER_LABEL_FONT_PATH,
            WatermarkConfig.CORNER_LABEL_FONT_SIZE,
        )
        left, _, right, _ = font.getbbox(WatermarkConfig.CORNER_LABEL_TEXT)
        width = right - left + 2 * WatermarkConfig.CORNER_LABEL_PADDING
        self.assertEqual(WatermarkConfig.CORNER_LABEL_FONT_SIZE, 28)
        self.assertEqual(WatermarkConfig.CORNER_LABEL_BORDER_WIDTH, 2)
        self.assertLessEqual(width + 2 * WatermarkConfig.CORNER_LABEL_MARGIN, 1024)

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

    def test_tiled_logo_opacity_reads_valid_value(self):
        with patch.dict(os.environ, {"WATERMARK_TILED_LOGO_OPACITY": "0.35"}):
            value = env_float("WATERMARK_TILED_LOGO_OPACITY", 0.35, 0.0, 1.0)
        self.assertEqual(value, 0.35)

    def test_tiled_logo_scale_reads_valid_value(self):
        with patch.dict(os.environ, {"WATERMARK_TILED_LOGO_SCALE": "1.5"}):
            value = env_float("WATERMARK_TILED_LOGO_SCALE", 1.0, 0.05, 3.0)
        self.assertEqual(value, 1.5)

    def test_tiled_logo_base_width_reads_valid_value(self):
        with patch.dict(os.environ, {"WATERMARK_TILED_LOGO_BASE_WIDTH": "200"}):
            value = round(
                env_float("WATERMARK_TILED_LOGO_BASE_WIDTH", 200.0, 1.0, 4000.0)
            )
        self.assertEqual(value, 200)

    def test_center_logo_opacity_reads_valid_value(self):
        with patch.dict(os.environ, {"WATERMARK_CENTER_LOGO_OPACITY": "0.4"}):
            value = env_float("WATERMARK_CENTER_LOGO_OPACITY", 0.55, 0.0, 1.0)
        self.assertEqual(value, 0.4)

    def test_image_logo_opacity_scales_existing_alpha(self):
        logo = Image.new("RGBA", (2, 1), (255, 255, 255, 255))
        logo.putalpha(Image.new("L", (2, 1), 128))
        scaled = _scale_alpha(logo, 0.35)
        self.assertEqual(scaled.getchannel("A").getpixel((0, 0)), 45)

    def test_uniform_diagonal_positions_are_even_and_rows_are_offset(self):
        positions = list(
            _uniform_tile_positions((500, 500), (100, 50), 20, 30, 0.5)
        )
        first_row = [x for x, y in positions if y == -50]
        second_row = [x for x, y in positions if y == 30]
        self.assertTrue(all(b - a == 120 for a, b in zip(first_row, first_row[1:])))
        self.assertTrue(all(b - a == 120 for a, b in zip(second_row, second_row[1:])))
        self.assertEqual(first_row[0] - second_row[0], 60)


if __name__ == "__main__":
    unittest.main()
