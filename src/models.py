import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size=32, output_size=32):
        super().__init__()
        
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
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(32),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.BatchNorm1d(32),
            ),
        ])

        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.fuse(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
