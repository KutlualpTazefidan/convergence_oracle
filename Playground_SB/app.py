import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data (you can replace this with your own dataset)
data = pd.DataFrame({
    'Year': [2000, 2010, 2020, 2030],
    'Psychology': [10, 15, 18, 20],
    'Medicine': [5, 8, 12, 15]
})

# Create a Linear Regression model
model = LinearRegression()

# Fit the model with the sample data
X = data[['Year']]
y = data['Psychology']
model.fit(X, y)

# Streamlit app
st.title("Linear Regression Prediction App")

# Sidebar
st.sidebar.header("Input Parameters")

# Year range selection
year_start, year_end = st.sidebar.slider("Year Range", 2000, 2030, (2000, 2030))

# Fields of Study selection
fields_of_study = st.sidebar.multiselect(
    "Fields of Study",
    data.columns[1:],  # Exclude the 'Year' column
    default=['Psychology']
)

# Display the selected input parameters
st.sidebar.write("Selected Input Parameters:")
st.sidebar.write(f"Year Range: {year_start} - {year_end}")
st.sidebar.write(f"Fields of Study: {', '.join(fields_of_study)}")

# Filter the data based on the selected parameters
filtered_data = data[(data['Year'] >= year_start) & (data['Year'] <= year_end)]

# Create input data for predictions
input_data = pd.DataFrame({'Year': filtered_data['Year']})
for field in fields_of_study:
    input_data[field] = 0  # Initialize with 0, you can modify this based on your input

# Predict using the model
predictions = model.predict(input_data[['Year']])

# Display predictions
st.write("Predictions:")
st.write(predictions)

# Display the original data
st.write("Original Data:")
st.write(filtered_data)

