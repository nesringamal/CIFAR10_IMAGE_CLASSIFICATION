# ==========================================================
# 🧠 CIFAR-10 - Image Prediction Script (Local Image Input)
# ==========================================================

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# ================== CIFAR-10 class names ==================
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ================== Device configuration ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== Define the CNN architecture ==================
# This must match exactly the model used during training
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # ================== Classifier ==================
        # Last Linear layer matches checkpoint: 2048 -> 256
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),  # Match checkpoint
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        # Flatten manually to match the Linear input
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ================== Load Best Model ==================
model_path = "models/best_model.pth"  # Make sure this path is correct
assert os.path.exists(model_path), "❌ Model file not found!"

model = EnhancedCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Best Model loaded successfully!")

# ================== Image preprocessing ==================
# Must match training transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 image size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# ================== Input image ==================
image_path = input("Enter image path (e.g. test_images/cat1.png): ").strip()
if not os.path.exists(image_path):
    print("❌ Image not found! Check the path.")
    exit()

image = Image.open(image_path).convert("RGB")
img_tensor = transform(image).unsqueeze(0).to(device)

# ================== Make prediction ==================
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = CLASSES[predicted.item()]

print(f"🔍 Predicted Class: **{predicted_class.upper()}**")
