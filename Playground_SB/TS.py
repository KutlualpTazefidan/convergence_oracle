import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Sidebar for user input
st.sidebar.header("User Input")

# Upload data file
data_file = st.sidebar.file_uploader("Upload Data CSV", type=["csv"])

if data_file is not None:
    # Read data from the uploaded CSV file
    input_data = pd.read_csv(data_file)

    # Title of the web app
    st.title("Prediction App")

    # Sidebar for user input
    st.sidebar.header("User Input")

    # Fields of Study
    fields_of_study = st.sidebar.multiselect(
        "Fields of Study",
        input_data.columns[1:],  # Exclude the "Year" column
        default=['Computer Science']
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

    # Load the machine learning model
    model = joblib.load('df.pkl')  # Replace 'your_model.pkl' with the actual model file path

    if st.button("Predict"):
        # Prepare the input data for prediction
        input_data_for_prediction = input_data[fields_of_study]

        # Make predictions using the loaded model
        predictions = model.predict(input_data_for_prediction)

        # Display predictions
        st.write("Predicted Values:")
        st.write(predictions)