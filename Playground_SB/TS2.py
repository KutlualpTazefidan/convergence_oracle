import joblib

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load your data containing the fields of study
data = pd.read_csv('cleaned_vectorized_data.csv')

# Explode the 'fieldsOfStudy' column to separate rows
data_exploded = data.explode('fieldsOfStudy')

# Get the unique fields of study
fieldsOfStudy_all = data_exploded['fieldsOfStudy'].unique()

# Generate years from 2000 to 2030
st.sidebar.header("User Input")

# Year Range
year_start = st.sidebar.slider("Year Start", min_value=2000, max_value=2030, step=1, value=2000)
year_end = st.sidebar.slider("Year End", min_value=2000, max_value=2030, step=1, value=2030)

# Fields of Study Selection
st.sidebar.header("Fields of Study Selection")
fieldsOfStudy = st.sidebar.multiselect(
    "Select Fields of Study",
    fieldsOfStudy_all.tolist(),
    default=fieldsOfStudy_all[:3]  # Default selection (you can change this)
)

# Filter data based on selected fields of study
filtered_data = data_exploded[data_exploded['fieldsOfStudy'].isin(fieldsOfStudy)]

# Title of the web app
st.title("Time Series Prediction App")

# Display selected fields of study
st.sidebar.header("Selected Fields of Study")
st.sidebar.write(fieldsOfStudy)

# Plot data for selected fields of study
plt.figure(figsize=(10, 6))
for field in fieldsOfStudy:
    field_data = filtered_data[filtered_data['fieldsOfStudy'] == field]
    plt.plot(field_data['year'], field_data['fieldsOfStudy'], label=field)

plt.xlabel("Year")
plt.ylabel("Fields of Study")
plt.title("Trends in Fields of Study Over Time")
plt.legend(loc="upper left")
st.pyplot(plt)

# Load XGBoost model
model = joblib.load('df.pkl')  # Replace with your model filename

if st.button("Predict"):

# Prepare input data for prediction (filtered_data contains only selected fields)
    selected_columns = ['year'] + fieldsOfStudy
    input_data_for_prediction = filtered_data[selected_columns].copy()

    # Make predictions
    predictions = model.predict(input_data_for_prediction)

    # Display predictions
    st.sidebar.header("Predicted Values")
    st.sidebar.write(predictions)
