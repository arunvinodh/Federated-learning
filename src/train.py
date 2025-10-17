import torch
import torch.nn as nn
import torch.optim as optim
from models import CNNDPLSTM  # <-- Import from model.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from opacus import PrivacyEngine
# Set data directory
data_dir = r"D:\Research Work\ArunPHDImplementation\FL\code\data"

# Define transforms (you can adjust resize if needed)
transform = transforms.Compose([
    transforms.Grayscale(),           # if your images are not grayscale
    transforms.Resize((28, 28)),      # adjust if needed
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Calculate split sizes
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

# Split dataset
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# To verify
print(f"Total images: {total_size}, Train: {len(train_dataset)}, Test: {len(test_dataset)}")
print(f"Number of classes: {len(full_dataset.classes)}, Class labels: {full_dataset.classes}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNDPLSTM(hidden_size=128, num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)
# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
