import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import flwr as fl
import torch
import numpy as np
from models.cnn_lstm import CNNDPLSTM
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from opacus import PrivacyEngine

class ASDClient(fl.client.NumPyClient):
    def __init__(self):
        print("[Client] Initializing ASDClient...")
        self.model = CNNDPLSTM(hidden_size=128, num_classes=8)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Define transforms
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        data_dir = r"D:\Research Work\ArunPHDImplementation\FL\code\data"
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

        train_size = int(0.8 * len(full_dataset))
        train_dataset, _ = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        print("[Client] ASDClient initialized and ready.")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_weights(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_weights(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Apply differential privacy
        privacy_engine = PrivacyEngine()
        self.model.train()
        self.model, optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0
        )

        self.model.train()
        for epoch in range(2):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
        print(f"[Client] Training complete with DP (ε = {epsilon:.2f}, δ = 1e-5)")

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_weights(parameters)
        self.model.eval()

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        data_dir = r"D:\Research Work\ArunPHDImplementation\FL\code\data"
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        train_size = int(0.8 * len(full_dataset))
        _, test_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = correct / total
        print(f"[Client] Evaluation - Accuracy: {accuracy:.4f}, Loss: {total_loss/total:.4f}")
        return total_loss / total, total, {"accuracy": accuracy}


if __name__ == "__main__":
    print("[Client] Connecting to server...")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=ASDClient())
    print("[Client] Disconnected.")
