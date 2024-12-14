import torch
from torch import nn
class SleepModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.model = self.build_model(num_features, num_classes)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)
        
    def build_model(self, num_features, num_classes):
        return nn.Sequential(
            nn.Linear(num_features, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.Linear(1000, 600),
            nn.ReLU(),
            nn.BatchNorm1d(600),
            # nn.Dropout(0.5),
            nn.Linear(600, num_classes)
    )

class GpaModel(nn.Module):
    def __init__(self, num_features,):
        super().__init__()
        self.model = self.build_model(num_features)
    
    def forward(self, x: torch.Tensor) -> torch.tensor:
        return self.model(x)
    
    def build_model(self, num_features):
        return nn.Sequential(
            nn.Linear(num_features, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
        )
        
from torch.utils.data import Dataset
class SleepDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return a tuple of input and target
        input_data = self.features[idx]
        target_data = self.targets[idx]
        return input_data, target_data