"""
Dataset Validation Script
-------------------------
This script checks the TrashNet folder structure,
counts images, detects corrupted files,
and records basic image information.
"""

from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


EXPECTED_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    """Validate the TrashNet dataset."""

    dataset_path = Path("data/raw/trashnet")

    print("=" * 60)
    print("TRASHNET DATASET VALIDATION")
    print("=" * 60)

    if not dataset_path.exists():
        print("ERROR: Dataset folder was not found.")
        return

    print(f"Dataset found at: {dataset_path}\n")

    total_images = 0
    corrupted_images = []
    image_dimensions = Counter()
    image_formats = Counter()

    print("Images per class:")
    print("-" * 60)

    for class_name in EXPECTED_CLASSES:
        class_path = dataset_path / class_name

        if not class_path.exists():
            print(f"{class_name:<12} MISSING")
            continue

        image_files = [
            file_path
            for file_path in class_path.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        image_count = len(image_files)
        total_images += image_count

        print(f"{class_name:<12} {image_count} images")

        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    width, height = image.size
                    image_format = image.format or "Unknown"

                    image_dimensions[(width, height)] += 1
                    image_formats[image_format] += 1

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as error:
                corrupted_images.append((image_path, str(error)))

    print("-" * 60)
    print(f"Total images: {total_images}")
    print(f"Corrupted images: {len(corrupted_images)}")

    print("\nMost common image dimensions:")
    print("-" * 60)

    for dimensions, count in image_dimensions.most_common(10):
        width, height = dimensions
        print(f"{width} x {height:<8} {count} images")

    print("\nImage formats:")
    print("-" * 60)

    for image_format, count in image_formats.items():
        print(f"{image_format:<12} {count} images")

    if corrupted_images:
        print("\nCorrupted image details:")
        print("-" * 60)

        for image_path, error_message in corrupted_images:
            print(f"{image_path}")
            print(f"Reason: {error_message}\n")
    else:
        print("\nNo corrupted images were found.")

    print("=" * 60)


if __name__ == "__main__":
    main()