from django import forms
from .models import SleepData

class SleepDataForm(forms.ModelForm):
    class Meta:
        model = SleepData
        fields = ['University_Year', 'Sleep_Duration', 'Study_Hours', 'Screen_Time', 
                 'Caffeine_Intake', 'Physical_Activity', 'Weekday_Sleep_Start', 
                 'Weekday_Sleep_End', 'Weekend_Sleep_Start', 'Weekend_Sleep_End']
        widgets = {
            'University_Year': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter your university year'
            }),
            'Sleep_Duration': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter sleep duration'
            }),
            'Study_Hours': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter study hours'
            }),
            'Screen_Time': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter screen time'
            }),
            'Caffeine_Intake': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter number of drinks'
            }),
            'Physical_Activity': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter physical activity minutes'
            }),
        }