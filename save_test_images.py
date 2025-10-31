import os
import random
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# ================== Define image transformations ==================
transform = transforms.Compose([
    transforms.ToTensor()
])

# ================== Load CIFAR-10 test dataset ==================
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# ================== CIFAR-10 class names ==================
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ================== Create main folder for test images ==================
base_dir = './test_images'
os.makedirs(base_dir, exist_ok=True)

# ================== Group indices by class ==================
class_indices = {cls: [] for cls in range(10)}
for idx, (_, label) in enumerate(testset):
    class_indices[label].append(idx)

# ================== Save 5 random images from each class ==================
for class_idx, img_indices in class_indices.items():
    class_name = classes[class_idx]
    class_dir = os.path.join(base_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # Pick 5 random samples from this class
    sample_indices = random.sample(img_indices, 5)

    for i, img_idx in enumerate(sample_indices):
        img, label = testset[img_idx]
        img = transforms.ToPILImage()(img)
        img_path = os.path.join(class_dir, f"{class_name}_{i}.png")
        img.save(img_path)

    print(f"✅ Saved 5 images for class '{class_name}'")

print("\n🎯 All images saved successfully in 'test_images/' folder!")
