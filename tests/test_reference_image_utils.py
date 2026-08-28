import base64
from io import BytesIO
import tempfile
import unittest

from PIL import Image

from app.utils.reference_image_utils import image_file_to_cropped_data_url


class ReferenceImageUtilsTests(unittest.TestCase):
    def test_crops_landscape_image_to_requested_portrait_ratio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = f"{temp_dir}/source.png"
            Image.new("RGB", (1200, 800), "red").save(image_path)

            data_url = image_file_to_cropped_data_url(image_path, "3:4")

            encoded = data_url.split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as result:
                self.assertEqual(result.size, (600, 800))

    def test_does_not_overwrite_source_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = f"{temp_dir}/source.png"
            Image.new("RGB", (900, 600), "blue").save(image_path)

            image_file_to_cropped_data_url(image_path, "1:1")

            with Image.open(image_path) as source:
                self.assertEqual(source.size, (900, 600))


if __name__ == "__main__":
    unittest.main()
