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
import matplotlib.pyplot as plt
from sklearn.utils import class_weight


#---Data Stuffs---
# Load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Drop specified columns
    columns_to_drop = ['Student_ID']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Explicitly map Stress_Level to numerical values: 0 for 'low', 1 for 'moderate', 2 for 'high'
    stress_level_mapping = {'High': 0, 'Moderate': 1, 'Low': 2}
    df['Stress_Level'] = df['Stress_Level'].map(stress_level_mapping)

    # Print out the mapping of stress levels
    print("Stress Level Classes Mapping:")
    for label, class_name in stress_level_mapping.items():
        print(f"Class {class_name}: {label}")

    # Extract features and set Stress_Level as the target
    features = df[['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 
                   'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA']].values
    target = df['Stress_Level'].values  # Stress_Level as the target



    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.long)

    return features_tensor, target_tensor

# Split into training and test sets
def create_datasets(features, target, test_size=0.3, val_size=0.1, batch_size=8):
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=test_size + val_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader



#---Main Neural Network Functions---
# Define MLP
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
    
# Early stopping and training with scheduler
def train_model(model, optimizer, train_loader, val_loader, epochs, patience, state_file):
    accuracy_metric = Accuracy(task='multiclass', num_classes=3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=epochs, steps_per_epoch=len(train_loader))
    history = []
    best_val_loss = float('inf')
    patience_counter = 0

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

        scheduler.step(val_loss)

        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss / len(train_loader),
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

        print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss / len(train_loader):.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_training_state(model, optimizer, epoch, history, state_file)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            save_training_state(model, optimizer, epoch, history, state_file)
            break

    return pd.DataFrame(history)


# Evaluation Function
def evaluate_model(model, data_loader):
    model.eval()
    accuracy_metric = Accuracy(task='multiclass', num_classes=3)
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



#---Saving the neural network for better training---
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



#Plotting the prediction of the model(line) vs the actual class value(points)
def plot_predictions_vs_actuals(model, data_loader):
    """
    Plots the model's predictions against the actual target values, with predictions on a line
    and targets as points.

    Parameters:
    - model: Trained PyTorch model
    - data_loader: DataLoader containing the dataset to evaluate

    Returns:
    - None
    """
    model.eval()
    predictions = []
    actuals = []

    # Collect predictions and actual values
    with torch.no_grad():
        for batch_features, batch_target in data_loader:
            outputs = model(batch_features)
            predicted_classes = torch.argmax(outputs, dim=1)
            predictions.extend(predicted_classes.cpu().numpy())
            actuals.extend(batch_target.cpu().numpy())

    # Convert to numpy arrays for visualization
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Sort actuals and predictions for better visualization
    sorted_indices = np.argsort(actuals)
    actuals_sorted = actuals[sorted_indices]
    predictions_sorted = predictions[sorted_indices]

    # Plot predictions as a line and actuals as points
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(predictions_sorted)), predictions_sorted, label="Predictions", color="blue", linestyle="-", linewidth=2)
    plt.scatter(range(len(actuals_sorted)), actuals_sorted, label="Actual Values", color="red", edgecolor="k", s=50)
    
    plt.xlabel("Data Points (sorted by Actual Values)")
    plt.ylabel("Sleep Quality Class")
    plt.title("Model Predictions vs Actual Values")
    plt.xticks([])  # Remove x-ticks for cleaner visualization
    plt.yticks(ticks=range(3), labels=[f"Class {i}" for i in range(3)])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
    
    
    
    
# Example Usage
if __name__ == "__main__":
    filepath = "student_lifestyle_dataset.csv"
    model_path = "ml_app/base/ml_code/mlp_classifier_state.pth"
    state_path = "ml_app/base/ml_code/training_state.pth"

    # Load and preprocess data
    features, target = load_and_preprocess_data(filepath)
    train_loader, val_loader, test_loader = create_datasets(features, target)

    # Initialize model, optimizer, and loss
    input_dim = features.shape[1]
    model = MLPClassifier(input_dim=input_dim, num_classes=3)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Load previous training state if available
    start_epoch, training_history = load_training_state(model, optimizer, state_path)

    # Train model with early stopping
    history = train_model(model, optimizer, train_loader, val_loader, epochs=1000, patience=5, state_file=state_path)

    # Save model state
    save_model_state(model, model_path)

    # Evaluate model
    model.load_state_dict(torch.load(model_path))
    test_loss, test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Analyze predictions
    predictions_df = analyze_predictions(model, test_loader)
    print(predictions_df.head())

    # Predict custom input
    scaler = StandardScaler()
    scaler.fit(features.numpy())
    # Example input (studying in hour, extracurricular in hours, sleep in hours, socializing in hours, physical activity in hours, GPA)
    sample_input = [5.3,3.5,8,4.2,3,2.75]
    prediction = predict_custom_input(model, scaler, sample_input)
    print(f"Predicted Stress Class: {prediction}")

    plot_predictions_vs_actuals(model, test_loader)