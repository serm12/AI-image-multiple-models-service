import base64
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.routers import api
from app.utils.watermark_utils import add_logo_watermark


class WatermarkSourceImmutabilityTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _data_url() -> str:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "white").save(buffer, "PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    async def test_existing_watermark_asset_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            logo_path = task_dir / "custom-watermark.png"
            Image.new("RGBA", (64, 20), (255, 255, 255, 128)).save(logo_path)
            before = hashlib.sha256(logo_path.read_bytes()).hexdigest()

            with (
                patch.object(api.WatermarkConfig, "LOGO_PATH", str(logo_path)),
                patch.object(api.task_manager, "update_task"),
                patch.object(api, "add_logo_watermark") as add_logo,
            ):
                result = await api.save_generated_image_outputs(
                    "test-task",
                    str(task_dir),
                    self._data_url(),
                    "reference.png",
                )

            self.assertEqual(before, hashlib.sha256(logo_path.read_bytes()).hexdigest())
            add_logo.assert_called_once()
            self.assertEqual(result, ["/taskfile/test-task/output_cropped_watermark.png"])

    async def test_real_watermark_render_keeps_source_asset_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            source = temp_dir / "source.png"
            output = temp_dir / "output.png"
            logo_path = Path("assets/logo_watermark-big_black_white.png")
            before = hashlib.sha256(logo_path.read_bytes()).hexdigest()
            Image.new("RGB", (512, 768), (90, 120, 150)).save(source)

            add_logo_watermark(
                str(source),
                str(output),
                logo_path=str(logo_path),
                resize_scale=1.0,
            )

            self.assertTrue(output.is_file())
            self.assertEqual(before, hashlib.sha256(logo_path.read_bytes()).hexdigest())

    async def test_missing_watermark_asset_fails_without_generating_text_logo(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            missing_logo = task_dir / "missing-watermark.png"

            with (
                patch.object(api.WatermarkConfig, "LOGO_PATH", str(missing_logo)),
                patch.object(api.task_manager, "update_task"),
                patch.object(api, "add_logo_watermark") as add_logo,
            ):
                with self.assertRaisesRegex(ValueError, "水印 Logo 文件不存在"):
                    await api.save_generated_image_outputs(
                        "test-task",
                        str(task_dir),
                        self._data_url(),
                        "reference.png",
                    )

            add_logo.assert_not_called()
            self.assertFalse(missing_logo.exists())


if __name__ == "__main__":
    unittest.main()
