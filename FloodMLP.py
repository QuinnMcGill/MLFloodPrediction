import torch.nn as nn

class FloodMLP(nn.Module):
    def __init__(self, input_size, dropout = 0.2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),  # Input size -> 64 neurons
            nn.BatchNorm1d(64),         # Reduces overfitting, speeds up learning, makes network less sensitive to feature scaling
            nn.ReLU(),                  # Add non-linearity
            nn.Dropout(dropout),

            nn.Linear(64, 32),          # 64 -> 32 neurons
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),           # 32 -> 1 neuron
            nn.Sigmoid()                # prob(flood = 1 | features)
        )

    def forward(self, x):
        return self.model(x)