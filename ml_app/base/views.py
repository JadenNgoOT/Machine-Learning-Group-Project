from django.shortcuts import render
from .forms import SleepDataForm
from .models import SleepData
from django.http import JsonResponse
import json
from .ml_code.main import prediction_api

def predict_sleep_quality(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        data = data.get('data')
        # print(data)
        prediction = prediction_api(data)
        print(prediction)
        return JsonResponse({"status" : "200",
                            "prediction": prediction
                            }, status=200)
        
    # This is a debugging message... Feel free to remove during deployment.
    # print("Inside the predict_sleep_quality") 
    
    return JsonResponse({"error" : "Invalid request"}, status=400)

def home(request):
    if request.method == 'POST':
        form = SleepDataForm(request.POST)
        if form.is_valid():
            sleep_data = form.save(commit=False)
            sleep_data.sleep_quality = predict_sleep_quality(sleep_data)
            sleep_data.save()
            return render(request, 'base/result.html', {'sleep_data': sleep_data})
    else:
        form = SleepDataForm()
    
    return render(request, 'base/home.html', {'form': form})

