from django import forms
from .models import SleepData

class SleepDataForm(forms.ModelForm):
    class Meta:
        model = SleepData
        fields = ['University_Year', 'Sleep_Duration', 'Study_Hours', 'Screen_Time', 
                  'Caffeine_Intake', 'Physical_Activity', 'Weekday_Sleep_Start', 'Weekend_Sleep_End']
        widgets = {
            'Weekday_Sleep_Start': forms.TimeInput(attrs={'type': 'time'}),
            'Weekend_Sleep_End': forms.TimeInput(attrs={'type': 'time'}),
        }

