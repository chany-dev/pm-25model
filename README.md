# PM2.5 Concentration Estimation using LightGBM Ensemble

This project predicts PM2.5 concentrations in Delhi (2020–2021) using satellite data, reanalysis data, and meteorological features. It employs a LightGBM-based ensemble model, optimized with Optuna, and saved as `.joblib` files for later inference.

---

## 📁 Project Structure

```
pm-25model/
│
├── model/
│   ├── savedmodels/                # Trained LightGBM models (joblib files)
│   ├── train_pm25_model.py         # Script to train and save ensemble models
│   ├── predict_pm25_input.py       # Script to predict PM2.5 using saved models
│   ├── Delhi_data.csv              # Cleaned dataset used for training and testing
│   └── requirements.txt            # Python dependencies
│
└── README.md                       # Project documentation
```

---

## 📌 Overview

- **Goal**: Estimate surface-level PM2.5 using AOD and meteorological features.
- **Approach**: Regression using LightGBM + Optuna tuning + Ensemble of models.
- **Data**: Cleaned and engineered dataset from 2020–2021.
- **Output**: Predicted PM2.5 concentration (µg/m³).

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r model/requirements.txt
```

### 2. Train the Model

This command trains an ensemble of 5 models and saves them to `savedmodels/`:

```bash
python model/trained.py
```

### 3. Predict PM2.5

Run this to predict PM2.5 interactively:

```bash
python model/predict.py
```

You'll be prompted to enter input features (e.g. PM10, NO2, Month_cat, etc.).

Sample output:

```
✅ PM2.5 Prediction Completed.
📈 Predicted PM2.5 Concentration: 134.72 µg/m³
```

---

## 🧠 Model Details

- **Model**: LightGBM Regressor
- **Tuning**: Optuna with TimeSeriesSplit CV
- **Ensemble**: 5 models with different seeds
- **Features**: Includes interaction terms like `T2M_PM10`, `NO2_WindSpeed`, etc.
- **Categorical Inputs**: Month_cat, Season, DayOfWeek_cat

---

## 📦 Requirements

Dependencies in `requirements.txt`:

```
lightgbm
optuna
numpy
pandas
seaborn
matplotlib
scikit-learn
joblib
```

Install using:

```bash
pip install -r model/requirements.txt
```

---

## 🏁 Contribution Context

Developed as part of **Bharatiya Antariksh Hackathon 2025 – Problem Statement 3**.
