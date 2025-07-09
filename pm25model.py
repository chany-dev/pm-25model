import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# 📥 Load and Clean Dataset
# ========================
df = pd.read_csv("Delhi_PM25_2020_2022_CLEANED_categorical_updated2.csv")


df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2021-12-31')].copy()

# Interpolate numeric missing values
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='forward')

# Categorical encoding
categorical_cols = ['Month_cat', 'Season', 'DayOfWeek_cat', 'IsHoliday']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# ========================
# 🧪 Train/Test Split
# ========================
df.sort_values('date', inplace=True)
train_df = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2021-06-30')]
test_df = df[(df['date'] >= '2021-07-01') & (df['date'] <= '2021-12-31')]

# ========================
# ✅ Feature Selection (Excluding lag/rolling)
# ========================
selected_features = [
    'T2M', 'U10M', 'V10M', 'WindSpeed', 'Temp_Diff',
    'RelativeHumidity', 'PS (Pa)', 'PBLTOP (Pa)', 'PBLH (m)',
    'Optical_Depth_047', 'LST_Day_1km',
    'Month_cat', 'Season', 'DayOfWeek_cat', 'IsHoliday'
]




# Derived interaction features (only these kept)
train_df.loc[:, 'T2M_PM10'] = train_df['T2M'] * train_df['PM10']
train_df.loc[:, 'NO2_WindSpeed'] = train_df['NO2'] * train_df['WindSpeed']
test_df.loc[:, 'T2M_PM10'] = test_df['T2M'] * test_df['PM10']
test_df.loc[:, 'NO2_WindSpeed'] = test_df['NO2'] * test_df['WindSpeed']


X_train = train_df[selected_features]
y_train = train_df['PM2.5']
X_test = test_df[selected_features]
y_test = test_df['PM2.5']

# ========================
# 🧪 Optuna Hyperparameter Tuning
# ========================
tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 5),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "seed": 42
    }

    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, random_state=42)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse")
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)

    return np.mean(scores)

# Run optimization
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("\n🏆 Best Parameters:", best_params)

# ========================
# 🤖 Ensemble Using Best Model
# ========================
seeds = [42, 101, 202, 303, 404]
ensemble_preds = []

print("\n🔁 Running Ensemble of Regression Models:")
for seed in seeds:
    model = lgb.LGBMRegressor(**best_params, objective='regression', random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    ensemble_preds.append(preds)

y_pred_ensemble = np.mean(ensemble_preds, axis=0)

# ========================
# 📊 Model Evaluation
# ========================
mae = mean_absolute_error(y_test, y_pred_ensemble)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
r2 = r2_score(y_test, y_pred_ensemble)

print("\n📊 Ensemble Model Performance:")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.2f}")

# ========================
# 📈 Feature Importance
# ========================
model = lgb.LGBMRegressor(**best_params, objective='regression', random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_

feature_imp_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🔍 Feature Importances:")
print(feature_imp_df.to_string(index=False))

print("\n📉 Least Important Features (Importance = 0):")
print(feature_imp_df[feature_imp_df['Importance'] == 0])

# Final evaluation
mae = mean_absolute_error(y_test, y_pred_ensemble)
mse = mean_squared_error(y_test, y_pred_ensemble)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_ensemble)

print("\n📊 Ensemble Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# ========================
# 📉 Plot Predictions vs Actual
# ========================
# plt.figure(figsize=(14, 6))
# plt.plot(y_test.values, label='Actual', color='black', linewidth=1)
# plt.plot(y_pred_ensemble, label='Predicted (Ensemble)', color='blue', linestyle='--')
# plt.title("PM2.5 Predictions (Regression Ensemble)")
# plt.xlabel("Time Index")
# plt.ylabel("PM2.5 (\u00b5g/m³)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # ========================
# # 📊 Plot Feature Importance
# # ========================
# plt.figure(figsize=(12, 6))
# sns.barplot(data=feature_imp_df, x='Importance', y='Feature', palette='viridis')
# plt.title("Feature Importances")
# plt.tight_layout()
# plt.show()

