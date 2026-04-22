import joblib
import pandas as pd

model = joblib.load("model.pkl")


columns = [
    'Marital status', 'Daytime/evening attendance',
    'Previous qualification', "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

# -------------------------------
# Sample Input
# -------------------------------

sample_data = [[ 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 20, 0,
    5, 5, 5, 5, 12.5, 0,
    5, 5, 5, 5, 12.5, 0,
    5.0, 2.5, 1.0
]]

sample_input = pd.DataFrame(sample_data, columns=columns)

print("Input shape:", sample_input.shape)

prediction = model.predict(sample_input)

if prediction[0] == 1:
    print("Student is likely to DROP OUT")
else:
    print("Student is likely to CONTINUE")