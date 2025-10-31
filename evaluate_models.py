import torch
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===========================
# Config
# ===========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = 'models'
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, 'best_model.pth')

# ===========================
# Load CIFAR-10 test dataset
# ===========================
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# ===========================
# Import model architectures (needed if using state_dict)
# ===========================
from train_cnn import CNN
from train_dense import DenseNet, DropoutNet

# ===========================
# Map filenames to model classes
# ===========================
model_class_map = {
    'cnn': CNN,
    'dense': DenseNet,
    'dropout': DropoutNet
}

# ===========================
# Evaluation function
# ===========================
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    return accuracy, all_labels, all_preds

# ===========================
# Detect if file is state_dict or whole model
# ===========================
def load_model(path, model_class=None):
    try:
        # Attempt to load as state_dict
        if model_class is None:
            raise ValueError("Model class needed for state_dict loading")
        model = model_class().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model
    except Exception:
        # If fails, try whole model
        model = torch.load(path, map_location=device)
        return model

# ===========================
# Evaluate all models
# ===========================
results = []

for file in os.listdir(model_dir):
    if file.endswith('.pth') and file != 'best_model.pth':
        file_lower = file.lower()
        model_class = None
        for key in model_class_map:
            if key in file_lower:
                model_class = model_class_map[key]
                break
        
        model_path = os.path.join(model_dir, file)
        model = load_model(model_path, model_class)
        model.to(device)
        model.eval()
        
        acc, labels, preds = evaluate(model, testloader)
        print(f'{file} accuracy: {acc:.2f}%')
        
        # Save confusion matrix
        cm = confusion_matrix(labels, preds)
        cm_path = os.path.join(results_dir, f'{file.replace(".pth","")}_confusion_matrix.png')
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {file}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(cm_path)
        plt.close()
        
        # Save results
        results.append({'Model': file, 'Accuracy': acc, 'Class': model_class.__name__ if model_class else 'WholeModel'})

# Save CSV report
df = pd.DataFrame(results)
csv_path = os.path.join(results_dir, 'model_comparison.csv')
df.to_csv(csv_path, index=False)
print(f'\nComparison report saved to {csv_path}')

# ===========================
# Select and save best model
# ===========================
if not df.empty:
    best_row = df.loc[df['Accuracy'].idxmax()]
    best_model_file = best_row['Model']
    # Reload best model
    file_lower = best_model_file.lower()
    model_class = None
    for key in model_class_map:
        if key in file_lower:
            model_class = model_class_map[key]
            break
    best_model_path_full = os.path.join(model_dir, best_model_file)
    best_model = load_model(best_model_path_full, model_class)
    torch.save(best_model.state_dict(), best_model_path)
    print(f'Best model: {best_model_file} with accuracy {best_row["Accuracy"]:.2f}%')
    print(f'Best model saved to {best_model_path}')
