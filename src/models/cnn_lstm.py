import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers import DPLSTM

class CNNDPLSTM(nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers=1):
        super(CNNDPLSTM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.flattened_size = None
        self.lstm = None  # Will initialize after inferring flattened_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        # Initialize LSTM once with inferred size
        if self.flattened_size is None:
            self.flattened_size = x.shape[1]
            self.lstm = DPLSTM(
                input_size=self.flattened_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True
            ).to(x.device)  # Important to place on correct device

        x = x.unsqueeze(1)  # (batch, seq_len=1, features)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
