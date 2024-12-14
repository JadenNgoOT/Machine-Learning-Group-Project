import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from Classes import SleepModel, SleepDataset, GpaModel
from tqdm import tqdm
from torchmetrics.classification import Accuracy
from sklearn.preprocessing import StandardScaler
import joblib

def train(epochs=50):
    num_classes = 10   # FLAGGER 1
    inputs, targets = get_synthethic_dataset()
    
    scaler = StandardScaler()
    inputs_scaled = scaler.fit_transform(inputs.numpy())  # Scale the inputs (for later use in prediction)
    inputs = torch.tensor(inputs_scaled, dtype=torch.float32)
    
    dataset = SleepDataset(inputs, targets)
    training_dataset, validation_dataset = random_split(dataset, (0.9, 0.1))
    
    model = SleepModel(inputs.size()[1], num_classes) 
    # Why Cross Entropy? Originall NLLLoss was intended but CrossEntropy implements it and also handles the logsoftmax.
    loss_fn = nn.CrossEntropyLoss()
    
    # Why Adam? Robust and used generally used for multiclass classification problems.
    # Also, Adam is used for datasets that ARE NOT large. 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5) 
    
    training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
    
    history = []
    losses = {
    'val_loss': [],
    'train_loss': [],
}
    train_accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes)
    validation_accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes)
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        # -- Training --
        for i, (inputs, targets) in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = loss_fn(outputs, targets)
            losses['train_loss'].append(loss)
                       
            running_loss += loss.item()
            
            # Backward pass: compute gradients
            loss.backward()
            
            train_accuracy_metric.update(outputs, targets)
            
            # Update the model parameters
            optimizer.step()
            
        # Calculate accuracy
        train_accuracy = train_accuracy_metric.compute()
        train_accuracy_metric.reset()
            
        # -- Validation Phase -- #
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.float(), targets.long()  # Ensure data types
                val_outputs = model(inputs)  # Forward pass
                val_loss = loss_fn(val_outputs, targets)  # Compute loss
                val_running_loss += val_loss.item()  # Accumulate validation loss
                losses['val_loss'].append(val_loss.item())  # Track validation loss

                validation_accuracy_metric.update(val_outputs, targets)  # Update validation accuracy metric
            
            val_accuracy = validation_accuracy_metric.compute() 
            validation_accuracy_metric.reset()
            
        metrics = {
            'epoch': epoch + 1,
            'loss': running_loss / len(training_dataloader),  # Average loss per batch in the epoch
            'acc': train_accuracy.item(),  # Accuracy in this epoch
            'val_loss' : val_running_loss / len(val_dataloader),
            'val_acc' : val_accuracy.item(),
        }
        history.append(metrics)
        print(f"Epoch [{metrics['epoch']}/{epochs}], Loss: {metrics['loss']:.4f}, Accuracy: {metrics['acc']:.4f}, Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_acc']:.4f}")
        
    # UNCOMMENT THIS WHEN YOU'RE HAPPY WITH THE MODEL YOU'VE CREATED;
    # Happy with the model means you are confident it works well
    # Means it is not overfitting
    # Means "good" accuracy and loss
    
    
    # torch.save(model.state_dict(), 'sleep_model.pth')
    # joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
    return history

def train_gpa(epochs=100):    
    inputs, targets = get_real_dataset()

    dataset = SleepDataset(inputs, targets)
    training_dataset, validation_dataset = random_split(dataset, (0.9, 0.1))

    model = GpaModel(inputs.size()[1])

    # Why MSE? GpaModel generates a line.
    loss_fn = nn.MSELoss()
    
    # Why Adam? Robust and used generally used for multiclass classification problems.
    # Also, Adam is used for datasets that ARE NOT large. 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    training_dataloader = DataLoader(training_dataset, batch_size=100, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=100, shuffle=False)
    
    history = []
    losses = {
    'val_loss': [],
    'train_loss': [],
}

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        # -- Training --
        for i, (inputs, targets) in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = loss_fn(outputs, targets)
            losses['train_loss'].append(loss)
                     
            running_loss += loss.item()
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            
        # -- Validation Phase -- #
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.float(), targets.long()  # Ensure data types
                val_outputs = model(inputs)  # Forward pass
                val_loss = loss_fn(val_outputs, targets)  # Compute loss
                val_running_loss += val_loss.item()  # Accumulate validation loss
                losses['val_loss'].append(val_loss.item())  # Track validation loss

            
        metrics = {
            'epoch': epoch + 1,
            'loss': running_loss / len(training_dataloader),
            'val_loss' : val_running_loss / len(val_dataloader),
        }
        history.append(metrics)
        print(f"Epoch [{metrics['epoch']}/{epochs}], Loss: {metrics['loss']:.4f}")
    
    return history

def get_synthethic_dataset():
    file_path = "student_sleep_patterns.csv"
    df = pd.read_csv(file_path)

    # Idea: Predict the gender of the students with a gender != Male || Female.
    # different = df[(df['Gender'] != "Male") & (df['Gender'] != "Female")]
    # print(len(different))

    # -- Cleaning -- #
    df["University_Year"] = df['University_Year'].str.replace(r'\D', '', regex=True).astype(int)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
    # df = df[df['Gender'].isin([0, 1])]
    df['Sleep_Quality'] = df['Sleep_Quality'] - 1


    # -- Torch Code -- #
    inputs = df[['Age',
            'University_Year',
            'Sleep_Duration',
            'Study_Hours',
            'Screen_Time',
            'Caffeine_Intake',
            'Physical_Activity',
            'Gender',]].values

    # scaler = StandardScaler()
    # inputs = scaler.fit_transform(inputs)

    inputs = torch.tensor(inputs, dtype=torch.float32)

    targets = torch.tensor(
        df['Sleep_Quality'].to_numpy(),
        dtype=torch.int64
    )

    return inputs, targets


def get_real_dataset():
    filepath = "student_lifestyle_dataset.csv"
    df = pd.read_csv(filepath)
    
    # -- Cleaning -- #
    df["Stress_Level"] = df['Stress_Level'].map(
        {'Low' : 1,
        'Moderate' : 2,
        'High': 3}
    )
    
    inputs = torch.tensor(
        df[[
            "Study_Hours_Per_Day",
            "Extracurricular_Hours_Per_Day",
            "Sleep_Hours_Per_Day",
            "Social_Hours_Per_Day",
            "Physical_Activity_Hours_Per_Day",
            "Stress_Level"]].to_numpy(),
        dtype=torch.float32
    )
    
    targets = torch.tensor(
        df["GPA"].to_numpy(),
        dtype=torch.float32
    )
    
    return inputs, targets

import matplotlib.pyplot as plt

def plot_loss(history):
    # Extract the loss values from the history
    train_loss = [epoch['loss'] for epoch in history]
    val_loss = [epoch['val_loss'] for epoch in history]
    epochs = range(1, len(train_loss) + 1)

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')

    # Add labels, legend, and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_my_value(input_data, model, scaler):
    """
    Function to predict a value based on the trained model and given input data.
    Assumes input_data is a 1D tensor or list that needs to be preprocessed similarly to training data.
    """
    # Apply the same preprocessing as during training (StandardScaler)
    input_data = torch.tensor(input_data, dtype=torch.float32).reshape(1, -1)  # Ensure input shape is [1, num_features]
    
    # Transform input data using the scaler
    input_data = scaler.transform(input_data.numpy())  # Using the same scaler as during training
    input_data = torch.tensor(input_data, dtype=torch.float32)  # Convert back to tensor
    
    # Set model to evaluation mode (important for inference)
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(input_data)  # Forward pass through the model

    # If it's a classification model, apply softmax and take the argmax for the predicted class
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item()  # Return the predicted class as a Python number

# -- Sleep Qual -- #
# history = train()
# plot_loss(history)

# -- GPA Stuff -- # 
# history = train_gpa(epochs=10)
# plot_loss(history)


# # Example input data for prediction
# Age, UniYear, Sleep (hrs), Study (hrs), Screen Time (hrs), Caffeine beverages, Physical Activity (mins)
# input_data = [22, 4, 8, 6, 4, 1, 20]

# Age, UniYear, Sleep (hrs), Study (hrs), Screen Time (hrs), Caffeine beverages, Physical Activity (mins), Gender (M:0 | F:1 | O:2)
# input_data = [21, 4, 8, 1, 6, 1, 90, 0]
input_data = [24, 4, 7, 6, 6, 1, 30, 1]
model = SleepModel(num_features=len(input_data), num_classes=10)  # FLAGGER 2 - 
model.load_state_dict(torch.load('sleep_model.pth'))  # Load the trained model
model.eval()  # Set to evaluation mode

# # Load the scaler
scaler = joblib.load('scaler.pkl')

predicted_class = predict_my_value(input_data, model, scaler)
print(f"Predicted class: {predicted_class}")