import torch.nn as nn
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),   # Batch Normalization
            nn.ReLU(),             # Activation Function
            nn.Dropout(0.5),       # Dropout for regularization
            
            # Hidden layer 1
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Hidden layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),       # Slightly reduced dropout
            
            # Output layer
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class ConvClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super(ConvClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        return self.conv(x)