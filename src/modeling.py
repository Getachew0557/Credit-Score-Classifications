import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
sys.path.append(os.path.abspath('../src'))
from model_evaluation import evaluate_model

# Function to train and evaluate Logistic Regression
def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    log_reg_preds_train = log_reg.predict(X_train)
    log_reg_preds_test = log_reg.predict(X_test)


    evaluate_model(y_train, log_reg_preds_train, "Logistic Regression (Train)")
    evaluate_model(y_test, log_reg_preds_test, "Logistic Regression (Test)")

# Function to train and evaluate Random Forest
def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_preds_train = rf_model.predict(X_train)
    rf_preds_test = rf_model.predict(X_test)

    evaluate_model(y_train, rf_preds_train, "Random Forest (Train)")
    evaluate_model(y_test, rf_preds_test, "Random Forest (Test)")

# Function to train and evaluate XGBoost
def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)

    evaluate_model(y_train, xgb_train_pred, "XGBoost (Train)")
    evaluate_model(y_test, xgb_test_pred, "XGBoost (Test)")

# Function to train and evaluate AdaBoost
def train_and_evaluate_adaboost(X_train, X_test, y_train, y_test):
    ada_model = AdaBoostClassifier()
    ada_model.fit(X_train, y_train)

    ada_preds_train = ada_model.predict(X_train)
    ada_preds_test = ada_model.predict(X_test)

    evaluate_model(y_train, ada_preds_train, "AdaBoost (Train)")
    evaluate_model(y_test, ada_preds_test, "AdaBoost (Test)")

# Function to train and evaluate Decision Tree
def train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    dt_preds_train = dt_model.predict(X_train)
    dt_preds_test = dt_model.predict(X_test)

    evaluate_model(y_train, dt_preds_train, "Decision Tree (Train)")
    evaluate_model(y_test, dt_preds_test, "Decision Tree (Test)")
