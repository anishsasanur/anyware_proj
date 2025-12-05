import torch
import numpy as np
from PIL import Image
import io
import base64
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Pydantic Models
class MaskData(BaseModel):
    index: int
    mask_base64: str
    format: str

class SegmentationResponse(BaseModel):
    masks: List[MaskData]
    count: int

# Global variables to store the model and processor
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("Loading SAM3 model...")
    try:
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        ml_models["processor"] = processor
        print("SAM3 model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(
    image: UploadFile = File(...),
    prompt: str = Form("Block")
):
    processor = ml_models.get("processor")
    if not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image file
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Set image
        inference_state = processor.set_image(pil_image)

        # Prompt model
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        # Get masks
        masks = output["masks"] # (N, H, W) boolean tensor
        
        results = []
        for i, mask in enumerate(masks):
            # Convert mask to binary image (0 or 255)
            mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_np)
            
            # Convert to base64
            buffered = io.BytesIO()
            mask_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            results.append(MaskData(
                index=i,
                mask_base64=img_str,
                format="png"
            ))

        return SegmentationResponse(masks=results, count=len(results))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)