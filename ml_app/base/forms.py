from django import forms
from .models import SleepData
from django.db import models
class SleepDataForm(forms.ModelForm):
    class Meta:
        model = SleepData
        fields = ['University_Year', 'Sleep_Duration', 'Study_Hours', 'Screen_Time', 
                 'Caffeine_Intake', 'Physical_Activity', 'Age', 'Gender']
        widgets = {
            'Age': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter age'
            }),
            'Gender': forms.Select(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Select your Gender'
            }),
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