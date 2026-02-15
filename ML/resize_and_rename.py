import os
from PIL import Image

# ===== SETTINGS =====
input_folder = "raw"        # folder with original photos
output_folder = "resized" # folder to save resized photos
size = (224, 224)           # AI-friendly size

os.makedirs(output_folder, exist_ok=True)

count = 1

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
        try:
            path = os.path.join(input_folder, filename)

            # Open image
            img = Image.open(path)

            # Convert to RGB (important for ML)
            img = img.convert("RGB")

            # Resize
            img = img.resize(size)

            # New filename
            new_name = f"img_{count}.jpg"
            save_path = os.path.join(output_folder, new_name)

            img.save(save_path, "JPEG", quality=95)

            print(f"Saved {new_name}")

            count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Done!")
