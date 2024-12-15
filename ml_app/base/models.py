from django.db import models

class SleepData(models.Model):
    Study_Hours_Per_Day = models.FloatField()
    Extracurricular_Hours_Per_Day = models.FloatField()
    Sleep_Hours_Per_Day = models.FloatField()
    Social_Hours_Per_Day = models.FloatField()
    Physical_Activity_Hours_Per_Day = models.FloatField()
    GPA = models.FloatField()
    Stress_Level = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Stress Study Data (GPA: {self.GPA})"
