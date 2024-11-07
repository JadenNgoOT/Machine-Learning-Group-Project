#Change nomclature later idgaf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    Dataset
)

df = pd.read_csv("Machine-Learning-Group-Project/student_sleep_patterns.csv")

def make_dataset(df: pd.DataFrame) -> Dataset:
    # Instantiate LabelEncoder for each categorical column
    gender_encoder = LabelEncoder()
    year_encoder = LabelEncoder()
    
    # Encode the 'Gender' and 'University_Year' columns
    df['Gender'] = gender_encoder.fit_transform(df['Gender'])
    df['University_Year'] = year_encoder.fit_transform(df['University_Year'])
    
    # Extract features and target columns
    features = df[['Age', 'Gender', 'University_Year', 'Study_Hours', 'Screen_Time', 'Caffeine_Intake', 'Physical_Activity', 'Sleep_Duration']].values
    target = df['Sleep_Quality'].values
    
    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.int64)
    
    # Return as TensorDataset
    return TensorDataset(features_tensor, target_tensor)

    
train_dataset = make_dataset(df)
for i in range(5):  # Print the first 5 samples
    features, target = train_dataset[i]
    print(f"Sample {i}: Features: {features}, Target: {target}")

    
