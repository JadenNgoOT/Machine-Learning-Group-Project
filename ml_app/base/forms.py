from django import forms
from .models import SleepData
from django.db import models
class SleepDataForm(forms.ModelForm):
    class Meta:
        model = SleepData
        fields = ['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 
                  'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA']
        widgets = {
            'Study_Hours_Per_Day': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter number of study hours per day'
            }),
            'Extracurricular_Hours_Per_Day': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter number of extracurricular hours per day'
            }),
            'Sleep_Hours_Per_Day': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter number of sleep hours per day'
            }),
            'Social_Hours_Per_Day': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter number of social hours per day'
            }),
            'Physical_Activity_Hours_Per_Day': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter number of physical activity hours per day'
            }),
            'GPA': forms.NumberInput(attrs={
                'class': 'form-control rounded-pill',
                'placeholder': 'Enter GPA'
            }),
        }