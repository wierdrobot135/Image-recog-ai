import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# -----------------------------
# 1. Paths
# -----------------------------
data_dir = "../resized"  # folder with class subfolders
save_path = "./food_model.pth"

# -----------------------------
# 2. Classes (auto-detect from folder)
# -----------------------------
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print("Classes detected:", classes)

# -----------------------------
# 3. Transform and Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# -----------------------------
# 4. Model
# -----------------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))

# -----------------------------
# 5. Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 6. Training loop
# -----------------------------
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")

# -----------------------------
# 7. Save model
# -----------------------------
torch.save({
    "model_state": model.state_dict(),
    "classes": classes
}, save_path)

print("Training finished! Model saved to:", save_path)
