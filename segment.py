import torch
import os
import numpy as np
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("data/jenga.png").convert("RGB")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="Block")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# Create output directory
output_dir = "output_masks"
os.makedirs(output_dir, exist_ok=True)

# Save masks
for i, mask in enumerate(masks):
    # mask is (1, H, W) boolean tensor
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_np)
    mask_path = os.path.join(output_dir, f"mask_{i}.png")
    mask_image.save(mask_path)
    print(f"Saved mask to {mask_path}")

print('Done!')
