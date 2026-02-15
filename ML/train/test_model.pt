import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# -----------------------------
# 1. Paths
# -----------------------------
model_path = "./food_model.pth"
test_dir = "./Test_images"  # folder with images to test

# -----------------------------
# 2. Load saved model
# -----------------------------
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
classes = checkpoint.get("classes", [])

# -----------------------------
# 3. Recreate model architecture
# -----------------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -----------------------------
# 4. Transform for input images
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# 5. Run predictions
# -----------------------------
for img_name in os.listdir(test_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    print(f"{img_name}: {classes[predicted.item()]} (confidence: {conf.item():.2f})")
