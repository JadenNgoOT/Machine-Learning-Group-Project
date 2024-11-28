# #Change nomclature later idgaf
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# import torch
# from torch.utils.data import (
#     DataLoader,
#     TensorDataset,
#     Dataset
# )

# df = pd.read_csv("./student_sleep_patterns.csv")

# def make_dataset(df: pd.DataFrame) -> Dataset:
#     # Instantiate LabelEncoder for each categorical column
#     gender_encoder = LabelEncoder()
#     year_encoder = LabelEncoder()
    
#     # Encode the 'Gender' and 'University_Year' columns
#     df['Gender'] = gender_encoder.fit_transform(df['Gender'])
#     df['University_Year'] = year_encoder.fit_transform(df['University_Year'])
    
#     # Extract features and target columns
#     features = df[['Age', 'Gender', 'University_Year', 'Study_Hours', 'Screen_Time', 'Caffeine_Intake', 'Physical_Activity', 'Sleep_Duration']].values
#     target = df['Sleep_Quality'].values
    
#     # Convert to tensors
#     features_tensor = torch.tensor(features, dtype=torch.float32)
#     target_tensor = torch.tensor(target, dtype=torch.int64)
    
#     # Return as TensorDataset
#     return TensorDataset(features_tensor, target_tensor)

    
# # train_dataset = make_dataset(df)
# # for i in range(5):  # Print the first 5 samples
# #     features, target = train_dataset[i]
# #     print(f"Sample {i}: Features: {features}, Target: {target}")

# print(df)


























import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

# Load and preprocess data
df = pd.read_csv("./student_sleep_patterns.csv")

# Drop weekend-related columns
columns_to_drop = [col for col in df.columns if "Weekend" in col]
df = df.drop(columns=columns_to_drop, errors='ignore')

# Encode categorical data
gender_encoder = LabelEncoder()
year_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender'])
df['University_Year'] = year_encoder.fit_transform(df['University_Year'])

# Extract features and target
features = df[['Age', 'Gender', 'University_Year', 'Study_Hours', 'Screen_Time', 
               'Caffeine_Intake', 'Physical_Activity', 'Sleep_Duration']].values
target = df['Sleep_Quality'].values

# Convert target to discrete classes (e.g., 1-3 -> 0, 4-7 -> 1, 8-10 -> 2)
target = np.digitize(target, bins=[3, 7])  # Create bins for 3 classes

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert to tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.long)  # Use long for classification

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_tensor, target_tensor, test_size=0.2, random_state=42)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define Models for Classification
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class ConvClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
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

# Training Function with Accuracy Metric
def train_model(model: nn.Module, optimizer: optim.Optimizer, dataloader: DataLoader, epochs: int):
    accuracy_metric = Accuracy(task='multiclass', num_classes=3)
    history = []  # To store loss and accuracy per epoch

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()  # Set model to training mode
        for batch_features, batch_target in dataloader:
            optimizer.zero_grad()  # Zero gradients before backward pass
            outputs = model(batch_features)  # Forward pass
            loss = F.cross_entropy(outputs, batch_target)  # Calculate loss using CrossEntropy

            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model weights

            epoch_loss += loss.item()  # Track loss for this epoch
            accuracy_metric.update(outputs, batch_target)  # Update accuracy metric

        # Calculate accuracy for this epoch
        accuracy = accuracy_metric.compute()

        metrics = {
            'epoch': epoch + 1,
            'loss': epoch_loss / len(dataloader),  # Average loss per batch in the epoch
            'acc': accuracy.item()  # Accuracy in this epoch
        }

        # Print metrics at regular intervals
        if (epoch + 1) % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {metrics['epoch']}/{epochs}: loss={metrics['loss']:.4f}, acc={metrics['acc']:.2f}")

        history.append(metrics)  # Add metrics to the history list

        # Reset accuracy metric for next epoch
        accuracy_metric.reset()

    return pd.DataFrame(history)  # Return metrics in DataFrame format

# Evaluation Function
def evaluate_model(model: nn.Module, test_loader: DataLoader):
    model.eval()  # Set model to evaluation mode
    accuracy_metric = Accuracy(task='multiclass', num_classes=3)
    test_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch_features, batch_target in test_loader:
            outputs = model(batch_features)  # Forward pass
            loss = F.cross_entropy(outputs, batch_target)  # Calculate loss
            test_loss += loss.item()  # Track loss over all batches

            accuracy_metric.update(outputs, batch_target)  # Update accuracy metric

            _, predicted = torch.max(outputs, 1)  # Get predicted class labels
            all_preds.extend(predicted.numpy())
            all_targets.extend(batch_target.numpy())

    # Calculate average loss and accuracy
    avg_loss = test_loss / len(test_loader)
    avg_acc = accuracy_metric.compute().item()  # Overall accuracy for the test set

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.2f}%")
    
    return pd.DataFrame({
        'loss': [avg_loss],
        'accuracy': [avg_acc]
    })

# Example usage:
model = MLPClassifier(input_dim=features_tensor.shape[1])
#model = LinearClassifier(input_dim=features_tensor.shape[1])
# model = ConvClassifier(input_dim=features_tensor.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
history_df = train_model(model, optimizer, train_loader, epochs=50)
evaluation_df = evaluate_model(model, test_loader)
