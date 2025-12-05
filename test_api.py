import requests
import base64
import os
import io
from PIL import Image

def test_segmentation():
    url = "https://f748e9c1fb58.ngrok-free.app/segment"
    image_path = "data/colored_boxes.png"
    output_dir = "outputs/test_output"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Sending request to {url} with image {image_path}...")
    
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"prompt": "Blocks"}
            response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print(f"Success! Received {result['count']} masks.")
            
            for mask_data in result['masks']:
                index = mask_data['index']
                b64_data = mask_data['mask_base64']
                
                # Decode and save
                img_data = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(img_data))
                
                save_path = os.path.join(output_dir, f"mask_{index}.png")
                img.save(save_path)
                print(f"Saved mask {index} to {save_path}")
                
        else:
            print(f"Error: Status code {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure 'python main.py' is running.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_segmentation()

