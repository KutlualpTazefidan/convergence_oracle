import streamlit as st
import json
import pandas as pd
import xgboost as xgb

# Load the model configuration from the JSON file
model_file_path = 'C:/Users/sa3id/spiced/CS/convergence_oracle/Playground_SB/Topified_vectorized_Science1900_2023/web/data/saved_models/yearly.json'

try:
    with open(model_file_path, 'r') as model_file:
        model_info = json.load(model_file)
    
    # Load the model 
    model = xgb.XGBRegressor()  

    st.title("Model Viewer")

    # Display the model hyperparameters and additional information as columns
    model_info["Model Type"] = type(model).__name__
    model_info["Number of Estimators"] = model.n_estimators  # Replace with the actual property
    model_info["max_depth"] = model.max_depth
    model_info["Scoring Metrics"] = "Mean Squared Error (MSE)"  # Replace with actual metrics
    model_info["Data Preprocessing Steps"] = "Standardization, Encoding"  # Replace with actual steps
    model_info["Model Performance ROC AUC"] = 0.95  # Replace with actual ROC AUC score

    df = pd.DataFrame(list(model_info.items()), columns=["Attribute", "Value"])
    st.dataframe(df)
except Exception as e:
    st.error(f"An error occurred: {e}")




st.write('---')
