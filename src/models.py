import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size=32, output_size=32):
        super().__init__()
        # Linear map from plaintext bits to ciphertext bits
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Module):
    def __init__(self, input_size=32, output_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

class CNNModel(nn.Module):
    def __init__(self, input_size=32, output_size=32):
        super().__init__()
        # Plaintext bits treated as a sequence of length 32
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        
    def forward(self, x):
        # x is (batch_size, 32)
        x = x.unsqueeze(1) # (batch_size, 1, 32)
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x
