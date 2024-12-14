from .ml_models import *
import joblib
import torch
from django.conf import settings
def prediction_api(input_data: dict): 
    # input_data is a dictionary, so we transform it into an ordered array to feed it into the model
    model_path = settings.BASE_DIR / "base" / "ml_code"
    data = build_ordered_array(input_data)

    model = MLPClassifier(input_dim=len(data), num_classes=5)
    model.load_state_dict(torch.load(model_path/"mlp_classifier_state.pth"))
    model.eval()
    
    scaler = joblib.load(model_path/"scaler.pkl")
    prediction = predict_custom_input(model, scaler, data)
    return prediction

def predict_custom_input(model, scaler, input_values):
    input_values = scaler.transform([input_values])
    input_tensor = torch.tensor(input_values, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class


"""
1. age
2. gender (1:male , 0: female)
3. uni year (0-3)
4. study (hours)
5. screen time (hours)
6. caffeine (# of drinks)
7. physical activity (mins)
8. sleep (hours)
"""
def build_ordered_array(data: dict):
    gender_dict = {
    "male": 1,
    "female": 0,
    "other": 2
    }
    data["Gender"] = gender_dict.get(data["Gender"], 2)  # -1 for invalid gender
    key_order = ["Age", "Gender", "University_Year", "Study_Hours", "Screen_Time", "Caffeine_Intake", "Physical_Activity", "Sleep_Duration"]
    ordered_values = [data[key] for key in key_order]
    return ordered_values