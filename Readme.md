# Stress Level Prediction using MLP Classifier

This project focuses on predicting student stress levels based on various lifestyle and academic factors using a **Multi-Layer Perceptron (MLP)** model implemented with PyTorch.

## Project Overview
The model predicts stress levels of students given the following features:
- **GPA**: Grade Point Average
- **Social Hours**: Time spent on social activities
- **Exercise Hours**: Time spent exercising
- **Study Hours**: Time dedicated to studying
- **Sleep Time**: Amount of sleep per day
- **Extracurricular**: Participation in extracurricular activities

The output is the predicted stress level (e.g., low, medium, or high).

## Model Architecture
The **MLPClassifier** is a neural network with the following architecture:

```python
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
```

### Key Features of the Model:
- **Input Layer**: Maps input features to a 512-dimensional space.
- **Hidden Layers**: Two hidden layers with 256 and 128 neurons, respectively.
- **Activation**: ReLU activation function for non-linearity.
- **Batch Normalization**: Applied after each linear transformation to stabilize training.
- **Dropout**: Dropout regularization (50% and 40%) to prevent overfitting.
- **Output Layer**: Outputs predictions for stress levels.

## Data
The dataset contains the following features:
| Feature              | Description                            |
|----------------------|----------------------------------------|
| GPA                  | Grade Point Average                   |
| Social Hours         | Hours spent on social activities      |
| Exercise Hours       | Hours spent exercising                |
| Study Hours          | Hours spent studying                  |
| Sleep Time           | Hours of sleep per day                |
| Extracurricular      | Participation in extracurriculars     |

The target variable is **Stress Level**, which is a classification problem (e.g., low, medium, high).
