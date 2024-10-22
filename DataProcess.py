import pandas as pd
import numpy as np

# Turns the CSV into a dataframe named stroke
stroke = pd.read_csv('healthcare-dataset-stroke-data - healthcare-dataset-stroke-data.csv')

# Drops IDs as it is non important for the ML
stroke = stroke.drop(['id'], axis = 1)

# Removing N/A from the BMI data and inputing the mean
Bmi_Mean = stroke['bmi'].mean()
print(Bmi_Mean)
stroke.replace(['N/A'], np.nan, inplace=False)
stroke['bmi'] = stroke['bmi'].fillna(Bmi_Mean, inplace = False)

#stroke['work_type'] = pd.Categorical(stroke['work_type'])
#print(stroke['work_type'].code)

