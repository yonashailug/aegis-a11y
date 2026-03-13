from typing import Any
import uuid

from PIL import Image
from pydantic import BaseModel
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor


# Re-defining the Pydantic schema from our Phase 1 specs
class ExtractedElement(BaseModel):
    element_id: str
    classification: str
    bounding_box: list[float]
    ocr_text: str
    html_tag: str


class LayoutDecomposer:
    def __init__(self, model_checkpoint: str = "microsoft/layoutlmv3-base"):
        """
        Loads the LayoutLMv3 processor and your fine-tuned model checkpoint.
        """
        # Note: You will replace "microsoft/layoutlmv3-base" with your fine-tuned
        # checkpoint directory containing the weights for your specific taxonomy
        # (headings, lists, tables, equations, diagrams).
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_checkpoint, apply_ocr=False
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_checkpoint)

        # Mapping model labels back to your HTML tags
        self.label_to_tag = {
            "heading": "<h2>",
            "paragraph": "<p>",
            "list": "<ul>",
            "table": "<table>",
            "equation": "<math>",
            "functional_diagram": "<figure>",
            "decorative_image": "<figure hidden>",
        }

    def decompose_image(
        self, image: Image.Image, ocr_data: dict[str, Any]
    ) -> list[ExtractedElement]:
        words = ocr_data["words"]
        boxes = ocr_data["boxes"]

        # Prepare inputs for the model
        encoding = self.processor(
            image, words, boxes=boxes, return_tensors="pt", truncation=True
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoding)

        # Get the predicted classifications
        logits = outputs.logits
        predictions = logits.argmax(-1).squeeze().tolist()
        token_boxes = encoding.bbox.squeeze().tolist()

        extracted_elements = []

        # Iterate through the predictions and map them to our Pydantic model
        # Note: This is a simplified token-to-word aggregation. In a production setting,
        # you will need to merge sub-word tokens belonging to the same bounding box region.
        for idx, pred in enumerate(predictions):
            # Skip special tokens (like [CLS], [SEP])
            if token_boxes[idx] == [0, 0, 0, 0]:
                continue

            predicted_label = self.model.config.id2label[pred]

            element = ExtractedElement(
                element_id=str(uuid.uuid4()),
                classification=predicted_label,
                bounding_box=token_boxes[idx],
                ocr_text=words[idx] if idx < len(words) else "",
                html_tag=self.label_to_tag.get(predicted_label, "<div>"),
            )
            extracted_elements.append(element)

        return extracted_elements
