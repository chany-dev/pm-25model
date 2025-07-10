# PM2.5 Concentration Estimation using LightGBM Ensemble

This project predicts PM2.5 concentrations in Delhi (2020–2021) using satellite data, reanalysis data, and meteorological features. It employs a LightGBM-based ensemble model, optimized with Optuna, and saved as `.joblib` files for later inference.

---

## Project Structure

pm-25model/
│
├── model/
│   ├── savedmodels/                                                     # Trained LightGBM models (joblib files)
│   ├── trained.py                                                       # Script to train and save ensemble models
│   ├── predict.py                                                       # Script to predict PM2.5 using saved models
│   ├── Delhi_PM25_2020_2022_CLEANED_categorical_updated2.csv           # Cleaned input data used for training and testing
│   └── requirements.txt                                                 # Python dependencies
│
└── README.md                                                            # This file

---

## Overview

- Goal: Estimate ground-level PM2.5 from satellite AOD and meteorological inputs.
- Approach: Regression using LightGBM + Optuna for tuning + Ensemble of 5 models.
- Data: Cleaned and feature-engineered dataset (delhidata.csv) from 2020–2021.
- Output: Predicted PM2.5 concentration (µg/m³).

---

## Setup Instructions

### 1. Install Dependencies

Run this command in your terminal:

    pip install -r model/requirements.txt

### 2. Train Model

This will train an ensemble of 5 models and save them to `model/savedmodels/`:

    python model/trained.py

### 3. Predict PM2.5 (Interactive)

Run this script and follow prompts to enter input features:

    python model/predict.py

Example user input:

    📥 Enter feature values for PM2.5 prediction:
    PM10: 180
    NO2: 45
    ...
    Month_cat (e.g. Jan, Feb...): Jan
    Season (Winter / Pre-monsoon / ...): Winter
    DayOfWeek_cat (Mon, Tue, ...): Tue

Example output:

    ✅ PM2.5 Prediction Completed.
    📈 Predicted PM2.5 Concentration: 134.72 µg/m³

---

## Requirements

Packages used in this project (listed in `requirements.txt`):

    lightgbm
    optuna
    numpy
    pandas
    seaborn
    matplotlib
    scikit-learn
    joblib

Install via pip:

    pip install -r model/requirements.txt

---

## Model Details

- Regressor: LightGBM
- Hyperparameter Tuning: Optuna (TPE sampler)
- Ensemble: 5 models trained with different random seeds
- Feature Engineering: Interaction terms like T2M_PM10, NO2_WindSpeed, etc.
- Categorical Inputs: Month_cat, Season, DayOfWeek_cat

---

## Contributions

This project was developed as part of the Bharatiya Antariksh Hackathon 2025 (Problem Statement 3).
