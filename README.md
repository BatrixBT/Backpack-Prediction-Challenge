# Backpack Weight Prediction using XGBoost & CatBoost

## **Overview**
This project is part of a Kaggle competition where the goal is to predict backpack weights based on given features. The solution leverages **XGBoost** and **CatBoost** for regression, along with hyperparameter tuning using **Optuna**.

## **Dataset**
The dataset is provided by the competition and consists of:
- `train.csv`: Training data with features and target labels.
- `test.csv`: Test data for which predictions need to be submitted.
- `training_extra.csv`: Additional training data that can be used for better model performance.

## **Approach**
The pipeline follows these key steps:
1. **Data Preprocessing:**
   - Handling missing values (if any)
   - Encoding categorical features using `OneHotEncoder`
   - Scaling numerical features using `StandardScaler`
2. **Modeling:**
   - Using `XGBoostRegressor` and `CatBoostRegressor` for prediction
   - Implementing a `VotingRegressor` for ensemble learning
3. **Hyperparameter Tuning:**
   - Optimizing hyperparameters using `Optuna`
4. **Model Evaluation:**
   - Using cross-validation (`cross_val_score`)
   - Evaluating performance using **Mean Squared Error (MSE)**

## **Results**
1. **Results:**
   - I finished top 1120 out of 3394 participants.

## **Dependencies**
To run this project, install the required libraries:
```bash
pip install numpy pandas scikit-learn xgboost catboost optuna
