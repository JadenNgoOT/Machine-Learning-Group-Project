from .ml_models import *
import joblib
import torch
from django.conf import settings

message_dict = {
    0: "Low Stress! Keep stress-free!",
    1: "Moderate Stress. Relax a bit.",
    2: "High Stress Level! It's time for a break..."
}

def prediction_api(input_data: dict): 
    # input_data is a dictionary, so we transform it into an ordered array to feed it into the model
    model_path = settings.BASE_DIR / "base" / "ml_code"
    data = build_ordered_array(input_data)

    model = MLPClassifier(input_dim=len(data), num_classes=3)
    model.load_state_dict(torch.load(model_path/"mlp_classifier_state.pth"))
    model.eval()
    
    scaler = joblib.load(model_path/"scaler.pkl")
    prediction = predict_custom_input(model, scaler, data)
    return message_dict[prediction]

def predict_custom_input(model, scaler, input_values):
    input_values = scaler.transform([input_values])
    input_tensor = torch.tensor(input_values, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class


"""
# Example input (studying in hour, extracurricular in hours, sleep in hours, socializing in hours, physical activity in hours, GPA)
1. Study_Hours_Per_Day
2. Extracurricular_Hours_Per_Day
3. Sleep_Hours_Per_Day
4. Social_Hours_Per_Day
5. Physical_Activity_Hours_Per_Day
6. GPA
"""
def build_ordered_array(data: dict):
    key_order = ["Study_Hours_Per_Day", "Extracurricular_Hours_Per_Day", "Sleep_Hours_Per_Day", "Social_Hours_Per_Day", "Physical_Activity_Hours_Per_Day", "GPA"]
    ordered_values = [data[key] for key in key_order]
    return ordered_values