import pandas as pd
import pickle
import os
import sys
# Append the correct src path for custom module imports
sys.path.append(os.path.abspath('../notebooks/model/'))
# Load the trained XGBoost model
def load_model():
    with open('../notebook/model/xgboost_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model
