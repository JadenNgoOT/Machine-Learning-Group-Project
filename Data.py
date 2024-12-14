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
import os
import joblib
# Load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Drop weekend-related columns
    columns_to_drop = [col for col in df.columns if "Weekend" in col]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Encode categorical data
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    df['University_Year'] = LabelEncoder().fit_transform(df['University_Year'])

    # Extract features and target
    features = df[['Age', 'Gender', 'University_Year', 'Study_Hours', 'Screen_Time', 
                   'Caffeine_Intake', 'Physical_Activity', 'Sleep_Duration']].values
    bins = [2, 4, 6, 8]
    bin_ranges = [f"Class {i}: {bins[i-1]+1 if i > 0 else '1'}-{bins[i] if i < len(bins) else '10'}" for i in range(len(bins)+1)]
    print("Class definitions:")
    for bin_range in bin_ranges:
        print(bin_range)  # Define bins for sleep quality classes
    target = np.digitize(df['Sleep_Quality'].values, bins=bins)  # Create bins for 5 classes
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    joblib.dump(scaler, "scaler.pkl")
    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.long)

    return features_tensor, target_tensor

# Split into training and test sets
def create_datasets(features, target, test_size=0.2, val_size=0.1, batch_size=16):
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=test_size + val_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

# Define Models for Classification
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
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

# Training Function with Validation
def train_model(model, optimizer, train_loader, val_loader, epochs, state_file=None):
    accuracy_metric = Accuracy(task='multiclass', num_classes=5)
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        accuracy_metric.reset()

        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = F.cross_entropy(outputs, batch_target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            accuracy_metric.update(outputs, batch_target)

        train_accuracy = accuracy_metric.compute().item()
        val_loss, val_accuracy = evaluate_model(model, val_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss / len(train_loader),
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

        # Save state periodically
        if state_file:
            save_training_state(model, optimizer, epoch, history, state_file)

        #print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss / len(train_loader):.4f}, "f"Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    return pd.DataFrame(history)

# Evaluation Function
def evaluate_model(model, data_loader):
    model.eval()
    accuracy_metric = Accuracy(task='multiclass', num_classes=5)
    test_loss = 0

    with torch.no_grad():
        for batch_features, batch_target in data_loader:
            outputs = model(batch_features)
            loss = F.cross_entropy(outputs, batch_target)
            test_loss += loss.item()
            accuracy_metric.update(outputs, batch_target)

    avg_loss = test_loss / len(data_loader)
    avg_acc = accuracy_metric.compute().item()

    return avg_loss, avg_acc

# Prediction Function
def analyze_predictions(model, data_loader):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for batch_features, batch_target in data_loader:
            outputs = model(batch_features)
            predicted_classes = torch.argmax(outputs, dim=1)
            predictions.extend(predicted_classes.cpu().numpy())
            actuals.extend(batch_target.cpu().numpy())

    return pd.DataFrame({'Actual': actuals, 'Predicted': predictions})

# Predict custom input
def predict_custom_input(model, scaler, input_values):
    input_values = scaler.transform([input_values])
    input_tensor = torch.tensor(input_values, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class

# Save model state
def save_model_state(model, filepath):
    torch.save(model.state_dict(), filepath)

# Load model state
def load_model_state(model, filepath):
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        print(f"Model state loaded from {filepath}")
    else:
        print(f"No saved model state found at {filepath}")

# Save training state
def save_training_state(model, optimizer, epoch, history, filepath):
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'history': history
    }
    torch.save(state, filepath)

# Load training state
def load_training_state(model, optimizer, filepath):
    if os.path.exists(filepath):
        state = torch.load(filepath)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        print(f"Training state loaded from {filepath}. Starting from epoch {state['epoch'] + 1}.")
        return state['epoch'], state['history']
    else:
        print(f"No saved training state found at {filepath}")
        return 0, []

def prediction_api(input_data):
    model = MLPClassifier(input_dim=len(input_data), num_classes=5)
    model.load_state_dict(torch.load('mlp_classifier_state.pth'))
    model.eval()
    
    scaler = joblib.load("scaler.pkl")
    prediction = predict_custom_input(model, scaler, input_data)
    return prediction
    
    
# Example Usage
if __name__ == "__main__":
    filepath = "./student_sleep_patterns.csv"
    model_path = "./mlp_classifier_state.pth"
    state_path = "./training_state.pth"

    # Load and preprocess data
    features, target = load_and_preprocess_data(filepath)
    train_loader, val_loader, test_loader = create_datasets(features, target)

    # Initialize model and optimizer
    input_dim = features.shape[1]
    model = MLPClassifier(input_dim=input_dim, num_classes=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load previous training state if available
    start_epoch, training_history = load_training_state(model, optimizer, state_path)

    # Train model
    history = train_model(model, optimizer, train_loader, val_loader, epochs=100, state_file=state_path)

    # Save model state
    save_model_state(model, model_path)

    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


    # Analyze predictions
    predictions_df = analyze_predictions(model, test_loader)
    print(predictions_df.head())

    # Predict custom input
    scaler = StandardScaler()
    scaler.fit(features.numpy())
    # Example input (age, gender (1: male, 0: female), uni year(0-3), study in hours, screen time in hours, caffine in # of drinks, physical activity in min, sleep in hours)
    sample_input = [22, 1, 3, 1, 8, 1, 120, 7]
    prediction = predict_custom_input(model, scaler, sample_input)
    print(f"Predicted Sleep Quality Class: {prediction}")

