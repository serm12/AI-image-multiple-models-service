from app.routers.api import get_public_watermark_files


def test_only_watermarked_images_are_public() -> None:
    files = [
        "/taskfile/example/output_cropped_original_14155662.png",
        "/taskfile/example/output_cropped_watermark.png",
        "/taskfile/example/output_reference_upscaled_2x.png",
        "/taskfile/example/output_reference_upscaled_2x_watermark.png",
        "/taskfile/example/params.json",
    ]

    assert get_public_watermark_files(files) == [
        "/taskfile/example/output_cropped_watermark.png",
        "/taskfile/example/output_reference_upscaled_2x_watermark.png",
    ]


def test_missing_files_returns_empty_list() -> None:
    assert get_public_watermark_files(None) == []
