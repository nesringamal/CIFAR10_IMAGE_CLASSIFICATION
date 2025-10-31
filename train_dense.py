# ==============================================
# 🧠 CIFAR-10 - Dense Neural Network Training
# ==============================================


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1️⃣ Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# 2️⃣ Define  Dense Neural Network (with BatchNorm + Dropout)
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.fc5 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

model = DenseNet().to(device)

# 3️⃣ Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler (reduce LR every 10 epochs)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# 4️⃣ Training
num_epochs = 30
train_acc_list = []

for epoch in range(num_epochs):
    model.train()
    correct, total = 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()
    train_acc = 100 * correct / total
    train_acc_list.append(train_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_acc:.2f}%")

# 5️⃣ Validation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_acc = 100 * correct / total
print(f"Validation Accuracy: {val_acc:.2f}%")

# 6️⃣ Save model
torch.save(model.state_dict(), "models/dense_model.pth")

# 7️⃣ Plot accuracy
plt.plot(train_acc_list, label="Train Accuracy")
plt.title(" Dense Network Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("results/dense_accuracy.png")
plt.close()
print("✅ Dense model trained and saved successfully!")
