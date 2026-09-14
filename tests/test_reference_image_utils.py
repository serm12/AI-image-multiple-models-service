import base64
from io import BytesIO
import tempfile
import unittest

from PIL import Image

from app.utils.reference_image_utils import image_file_to_fitted_data_url


class ReferenceImageUtilsTests(unittest.TestCase):
    def test_fits_landscape_image_to_requested_portrait_ratio_without_cropping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = f"{temp_dir}/source.png"
            source = Image.new("RGB", (1200, 800), "red")
            source.putpixel((0, 0), (0, 255, 0))
            source.putpixel((1199, 799), (0, 0, 255))
            source.save(image_path)

            data_url = image_file_to_fitted_data_url(image_path, "3:4")

            encoded = data_url.split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as result:
                self.assertEqual(result.size, (1200, 1600))
                self.assertEqual(result.getpixel((0, 400)), (0, 255, 0))
                self.assertEqual(result.getpixel((1199, 1199)), (0, 0, 255))

    def test_fits_narrow_portrait_by_extending_sides_without_cropping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = f"{temp_dir}/source.png"
            source = Image.new("RGB", (600, 1000), "beige")
            source.putpixel((0, 0), (255, 0, 0))
            source.putpixel((599, 999), (0, 255, 0))
            source.save(image_path)

            data_url = image_file_to_fitted_data_url(image_path, "3:4")

            encoded = data_url.split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as result:
                self.assertEqual(result.size, (750, 1000))
                self.assertEqual(result.getpixel((75, 0)), (255, 0, 0))
                self.assertEqual(result.getpixel((674, 999)), (0, 255, 0))

    def test_does_not_overwrite_source_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = f"{temp_dir}/source.png"
            Image.new("RGB", (900, 600), "blue").save(image_path)

            image_file_to_fitted_data_url(image_path, "1:1")

            with Image.open(image_path) as source:
                self.assertEqual(source.size, (900, 600))


if __name__ == "__main__":
    unittest.main()
