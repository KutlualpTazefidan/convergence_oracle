import streamlit as st
import pandas as pd
import plotly.express as px

# Sample data (you can replace this with your own dataset)
data = pd.DataFrame({
    'Year': [2000, 2010, 2020, 2030],
    'Mathematics': [10, 15, 18, 20],
    'Computer Science': [5, 8, 12, 15],
    'Physics': [8, 12, 15, 18],
    'Materials Science': [6, 10, 14, 16],
    'Medicine': [4, 6, 8, 10],
    'Psychology': [3, 5, 7, 9],
    'Chemistry': [7, 9, 11, 13],
    'Biology': [9, 11, 13, 15],
    'Business': [12, 14, 16, 18],
    'Economics': [14, 16, 18, 20],
    'Geography': [11, 13, 15, 17],
    'Geology': [10, 12, 14, 16],
    'Sociology': [5, 7, 9, 11],
    'Political Science': [4, 6, 8, 10],
    'History': [6, 8, 10, 12],
    'Philosophy': [8, 10, 12, 14],
    'Environmental Science': [7, 9, 11, 13],
    'Engineering': [9, 11, 13, 15]
})

# Streamlit app
st.title("Linear Regression Prediction App")

# Sidebar
st.sidebar.header("Input Parameters")

# Year range selection
year_start, year_end = st.sidebar.slider("Year Range", 2000, 2030, (2000, 2030))

# Fields of Study selection with modified default values
default_fields_of_study = ['Mathematics', 'Computer Science','Physics','Materials Science', 'Medicine','Psychology', 'Chemistry','Biology',
       'Business', 'Economics', 'Geography','Geology', 'Sociology', 'Political Science','History',
       'Philosophy', 'Environmental Science', 'Engineering']

fields_of_study = st.sidebar.multiselect(
    "Fields of Study",
    data.columns[1:],  # Exclude the 'Year' column
    default=default_fields_of_study  # Set the modified default values here
)

# Display the selected input parameters
st.sidebar.write("Selected Input Parameters:")
st.sidebar.write(f"Year Range: {year_start} - {year_end}")
st.sidebar.write(f"Fields of Study: {', '.join(fields_of_study)}")

# Filter the data based on the selected parameters
filtered_data = data[(data['Year'] >= year_start) & (data['Year'] <= year_end)]

# Create a line chart using Plotly Express
fig = px.line(filtered_data, x='Year', y=fields_of_study, title='Line Chart')
st.plotly_chart(fig)

# Display the original data
st.write("Original Data:")
st.write(filtered_data)

# Optionally, you can display additional information here
