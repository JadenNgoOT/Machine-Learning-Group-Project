from django.db import models

class SleepData(models.Model):
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other'),
    ]
    Age = models.IntegerField()
    Gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    University_Year = models.IntegerField()
    Sleep_Duration = models.FloatField()
    Study_Hours = models.FloatField()
    Screen_Time = models.FloatField()
    Caffeine_Intake = models.FloatField()
    Physical_Activity = models.FloatField()
    Sleep_Quality = models.FloatField(null=True, blank=True)
    Weekday_Sleep_Start = models.TimeField()
    Weekday_Sleep_End = models.TimeField()
    Weekend_Sleep_Start = models.TimeField()
    Weekend_Sleep_End = models.TimeField()

    def __str__(self):
        return f"Sleep Data for Year {self.University_Year}"

