from django.shortcuts import render
from .forms import SleepDataForm
from .models import SleepData

def predict_sleep_quality(sleep_data):
    # Placeholder function for prediction
    # In the future, replace this with your RNN model
    return (sleep_data.Sleep_Duration * 0.3 +
            sleep_data.Study_Hours * 0.1 +
            (24 - sleep_data.Screen_Time) * 0.2 +
            (10 - sleep_data.Caffeine_Intake) * 0.1 +
            sleep_data.Physical_Activity * 0.3) * 10

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

