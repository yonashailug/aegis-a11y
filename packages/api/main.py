from fastapi import FastAPI, HTTPException
from cv_layer import LayoutDecomposer, convert_pdf_to_images, extract_ocr_data

app = FastAPI(
    title="Aegis-a11y",
    description=""
)

decomposer = LayoutDecomposer("microsoft/layoutlmv3-base")

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Aegis-a11y"}

@app.post("/api/v1/decompose")
async def decompose_document():
    try:
        images = convert_pdf_to_images("../../docs/pdfs/Resume.pdf")
        document_elements = []

        for image in images:
            ocr_data = extract_ocr_data(image)
            page_elements = decomposer.decompose_image(image, ocr_data)
            document_elements.extend(page_elements)
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    print("Hello from api!")


if __name__ == "__main__":
    main()
