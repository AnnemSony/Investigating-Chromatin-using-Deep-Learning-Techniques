import os
import numpy as np
import torch
from PIL import Image
from transformers import SamProcessor, SamModel

model_path = "sam20modelhisE.pth"
image_folder = "test 11.1.22 - Copy"  # Replace with your image folder path
output_folder = "outputhis2"  # Replace with your output folder path

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Load the trained model
my_mito_model = SamModel.from_pretrained("facebook/sam-vit-base")
my_mito_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load model on CPU

# Set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_mito_model.to(device)
my_mito_model.eval()

# Iterate over images in the folder
for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, image_name)
        large_test_image = Image.open(image_path).convert("RGB")
        large_test_image_np = np.array(large_test_image)

        # Prepare the image for the model
        inputs = processor(large_test_image_np, return_tensors="pt")

        # Move the input tensor to the GPU if it's not already there
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = my_mito_model(**inputs, multimask_output=False)

        # Apply sigmoid to get probability map
        prob_map = torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze()

        # Convert soft mask to hard mask
        prediction = (prob_map > 0.5).astype(np.uint8)

        # Convert mask to image
        prediction_image = Image.fromarray(prediction * 255)  # Convert to binary image (0 or 255)

        # Resize mask to match original image size
        prediction_image = prediction_image.resize(large_test_image.size)

        # Convert mask to RGBA
        prediction_rgba = prediction_image.convert("RGBA")
        mask_data = prediction_rgba.getdata()
        new_mask_data = []
        for item in mask_data:
            # Change all white (255) pixels to red
            if item[0] == 255:
                new_mask_data.append((255, 0, 0, 128))  # Red color with transparency
            else:
                new_mask_data.append((0, 0, 0, 0))  # Transparent
        prediction_rgba.putdata(new_mask_data)

        # Overlay mask on the original image
        combined_image = Image.alpha_composite(large_test_image.convert("RGBA"), prediction_rgba)

        # Save the combined image
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_overlay.png")
        combined_image.save(output_path)

        print(f"Processed and saved: {output_path}")

print("Processing complete!")
