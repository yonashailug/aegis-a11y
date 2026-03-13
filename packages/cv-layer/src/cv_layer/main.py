from typing import Any

from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Ensure Tesseract v5.3.0 is installed on your system as per the Aegis-A11y baseline
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


def normalize_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    """
    Normalizes bounding box coordinates to a 0-1000 scale for LayoutLMv3.
    bbox format: [x0, y0, x1, y1]
    """
    return [
        int(1000 * (bbox[0] / width)),  # x0 normalized
        int(1000 * (bbox[1] / height)),  # y0 normalized
        int(1000 * (bbox[2] / width)),  # x1 normalized
        int(1000 * (bbox[3] / height)),  # y1 normalized
    ]


def extract_ocr_data(image: Image.Image) -> dict[str, Any]:
    """
    Runs Tesseract OCR to extract words and normalized bounding boxes.
    """
    width, height = image.size

    # Run OCR and get dictionary output
    ocr_df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    boxes = []

    for i in range(len(ocr_df["text"])):
        word = ocr_df["text"][i].strip()
        if word:  # Ignore empty strings
            x, y, w, h = (
                ocr_df["left"][i],
                ocr_df["top"][i],
                ocr_df["width"][i],
                ocr_df["height"][i],
            )
            # Tesseract returns (x, y, w, h). Convert to (x0, y0, x1, y1)
            bbox = [x, y, x + w, y + h]
            normalized_box = normalize_bbox(bbox, width, height)

            words.append(word)
            boxes.append(normalized_box)

    return {"words": words, "boxes": boxes}


def convert_pdf_to_images(pdf_path: str) -> list[Image.Image]:
    """Converts a PDF file to a list of PIL Images."""
    return convert_from_path(pdf_path)


def main():
    print("Hello from cv-layer!")


if __name__ == "__main__":
    main()
