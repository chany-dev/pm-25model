"""
 PM2.5 Prediction Script (Interactive)
----------------------------------------

This script predicts PM2.5 concentration based on user-input atmospheric features.
It uses an ensemble of LightGBM models trained previously (stored in /saved_models/).

 How It Works:
- Takes user input for features like PM10, NO2, T2M, etc.
- Validates categorical entries (e.g., Month, Season)
- Performs required feature engineering
- Loads all trained models and performs ensemble prediction

 Output: Predicted PM2.5 concentration (in µg/m³)

 Run this script to interactively test PM2.5 predictions from raw inputs.
"""


import joblib
import numpy as np
import pandas as pd
import os

# Define the order of features for input
input_features = [
    'PM10', 'NO2', 'SO2',
    'T2M', 'U10M', 'V10M', 'WindSpeed', 'Temp_Diff',
    'RelativeHumidity', 'PS (Pa)', 'PBLTOP (Pa)', 'PBLH (m)',
    'Optical_Depth_047', 'LST_Day_1km',
    'Month_cat', 'Season', 'DayOfWeek_cat'
]

# Define valid categorical values (used during training)
valid_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
valid_seasons = ['Winter', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']
valid_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Ask for input dynamically
print("\n📥 Enter feature values for PM2.5 prediction:")
user_input = {}
for feature in input_features:
    if feature == 'Month_cat':
        val = input(f"{feature} (e.g. Jan, Feb...): ")
        while val not in valid_months:
            val = input(f"❌ Invalid. Re-enter {feature} from {valid_months}: ")
    elif feature == 'Season':
        val = input(f"{feature} (Winter / Pre-monsoon / Monsoon / Post-monsoon): ")
        while val not in valid_seasons:
            val = input(f"❌ Invalid. Re-enter {feature} from {valid_seasons}: ")
    elif feature == 'DayOfWeek_cat':
        val = input(f"{feature} (Mon, Tue, ..., Sun): ")
        while val not in valid_days:
            val = input(f"❌ Invalid. Re-enter {feature} from {valid_days}: ")
    else:
        while True:
            try:
                val = float(input(f"{feature}: "))
                break
            except ValueError:
                print("❌ Please enter a valid number.")
    user_input[feature] = val

# ============================
# Feature Engineering
# ============================
df = pd.DataFrame([user_input])
df['Month_cat'] = df['Month_cat'].astype('category')
df['Season'] = df['Season'].astype('category')
df['DayOfWeek_cat'] = df['DayOfWeek_cat'].astype('category')

df['T2M_PM10'] = df['T2M'] * df['PM10']
df['NO2_WindSpeed'] = df['NO2'] * df['WindSpeed']
df['AOD_Wind'] = df['Optical_Depth_047'] * df['WindSpeed']
df['TDiff_RH'] = df['Temp_Diff'] * df['RelativeHumidity']

# Final feature set used in model
final_features = input_features + ['T2M_PM10', 'NO2_WindSpeed', 'AOD_Wind', 'TDiff_RH']
X = df[final_features]

# ============================
# Load Ensemble Models & Predict
# ============================
model_dir = "saved_models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
preds = []

for file in model_files:
    model_path = os.path.join(model_dir, file)
    model = joblib.load(model_path)
    pred = model.predict(X)[0]  # single prediction
    preds.append(pred)

# Average prediction
ensemble_prediction = np.mean(preds)

# ============================
# Output Prediction
# ============================
print("\n✅ PM2.5 Prediction Completed.")
print(f"📈 Predicted PM2.5 Concentration: {ensemble_prediction:.2f} µg/m³")
