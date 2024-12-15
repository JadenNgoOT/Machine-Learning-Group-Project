from Data import prediction_api
import pandas as pd
# Example input (age, gender (1: male, 0: female), uni year(0-3), study in hours, screen time in hours, caffine in # of drinks, physical activity in min, sleep in hours)
#
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
# print(prediction_api([22, 1, 3, 1, 8, 1, 120, 7]))

df = pd.read_csv("student_lifestyle_dataset.csv")
highest_gpa = df["GPA"].max()
print(highest_gpa)