import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib



# Generate years from 2000 to 2030
#years = list(range(2000, 2031))  # Includes years from 2000 to 2030
# Sidebar for user input
st.sidebar.header("User Input")

# Year Start
year_start = st.sidebar.slider("Year Start", min_value=2000, max_value=2030, step=1, value=2000)

# Year End
year_end = st.sidebar.slider("Year End", min_value=2000, max_value=2030, step=1, value=2030)

# Generate years based on user input
years = list(range(year_start, year_end + 1))
# Define all fields of study
fields_of_study_all = [
    'Mathematics', 'Computer Science', 'Physics', 'Materials Science',
    'Medicine', 'Psychology', 'Chemistry', 'Biology', 'Business',
    'Economics', 'Geography', 'Geology', 'Sociology', 'Political Science',
    'History', 'Philosophy', 'Environmental Science', 'Engineering']

# Create a dictionary with sample data for all fields of study
data = {'Year': years}
for field in fields_of_study_all:
    data[field] = np.random.randint(1, 100, len(years))

# Create a DataFrame from the data
input_data = pd.DataFrame(data)

# Title of the web app
st.title("Prediction App")

# Sidebar for user input
st.sidebar.header("User Input")

# Fields of Study
fields_of_study = st.sidebar.multiselect(
    "Fields of Study",
    fields_of_study_all,  # Use the list of all fields
    default=['Mathematics', 'Computer Science', 'Physics']
)

# Display selected fields of study
st.write("Selected Fields of Study:", fields_of_study)

# Plot data for selected fields of study
plt.figure(figsize=(10, 6))
for field in fields_of_study:
    plt.plot(input_data['Year'], input_data[field], label=field)

plt.xlabel("Year")
plt.ylabel("Value")
plt.title("Trends in Fields of Study Over Time")
plt.legend(loc="upper left")
st.pyplot(plt)

# Other app components and linear regression model can be added here

# model = joblib.load('trained_model_LR.pkl')

# if st.button("Predict"):
#     # Prepare the input data for prediction
#     input_data_for_prediction = input_data[fields_of_study]

#     # Make predictions
#     predictions = model.predict(input_data_for_prediction)

#     # Display predictions
#     st.write("Predicted Values:")
#     st.write(predictions)