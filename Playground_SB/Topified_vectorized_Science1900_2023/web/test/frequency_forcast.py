import pickle
import streamlit as st
import pandas as pd
import pickle
import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json
from xgboost import plot_importance
import datetime
import numpy as np
# load the model
#     model = pickle.load(file)
## PULL Functions from custom functions

aws_dfs=pd.read_csv('C:/Users/sa3id/spiced/test/convergence_oracle/Playground_SB/Topified_vectorized_Science1900_2023/web/data/streamlit_data.csv')
# Extract unique topic codes and topic lists from the DataFrame
unique_topics = aws_dfs[['topic_code', 'topic_list']].drop_duplicates()

# Create a dictionary mapping topic_code to topic_list
topic_mapping = dict(zip(unique_topics['topic_code'], unique_topics['topic_list']))
def plot_bertopic_data_frequency(selected_topic, topic_mapping):
    aws_dfs['ones_column'] = 1
    aws1_dfs = aws_dfs[['publicationDate', 'topic_code', 'citationCount', 'influentialCitationCount', 'topic_list','ones_column']]
    aws1_dfs['topic_code'] = pd.to_numeric(aws1_dfs['topic_code'], downcast='integer')
    aws1_dfs['citationCount'] = pd.to_numeric(aws1_dfs['citationCount'], downcast='integer')
    aws1_dfs = aws1_dfs[aws1_dfs['topic_code'] != -1]
    aws1_dfs = aws1_dfs.replace({None: np.nan})
    aws1_dfs = aws1_dfs.dropna(subset=['publicationDate'])  # Drop rows with missing publicationDate
    aws1_dfs['publicationDate'] = pd.to_datetime(aws1_dfs['publicationDate'], errors='coerce')  # Use 'coerce' to handle incorrect dates
    aws1_dfs['year_month'] = aws1_dfs['publicationDate'].dt.strftime('%Y-%m')
    aws1_dfs['year_month'] = pd.to_datetime(aws1_dfs['year_month'], format='%Y-%m')   
    grouped_df = aws1_dfs.groupby(['topic_code', 'year_month'])['ones_column'].sum().reset_index()
    #print(aws1_dfs)
    #grouped_df.to_csv('yearly_monthly_SL_data.csv',index=False)
    pivot_df = grouped_df.pivot(index='year_month', columns='topic_code', values='ones_column')

    pivot_df.fillna(0, inplace=True)
    pivot_df = pivot_df.reset_index()

    # Separating the training set and testing set
    train_data=pivot_df[pivot_df['year_month'].dt.year<2014].reset_index(drop = True)
    test_data=pivot_df[pivot_df['year_month'].dt.year>2013].reset_index(drop = True)

    # Define the target column (Topic you want to forecast)
    target_topic = selected_topic  # Change this to the Topic you want to forecast

    # Prepare the training and testing data
    X_train = train_data.drop(target_topic, axis=1)
    
    y_train = train_data[target_topic]  # Shift by 1 to align with next year's frequency
    
    X_test = test_data.drop(target_topic, axis=1)
    
    y_test = test_data[target_topic]

    # Shift the target column to align with next year's frequency
    y_train = train_data[target_topic].shift(-1).dropna()
    y_test = test_data[target_topic].shift(-1).dropna()

    # Exclude the 'YearMonth' column from the training and testing data
    X_train = train_data.drop(columns=['year_month', target_topic]).iloc[:-1]
    X_test = test_data.drop(columns=['year_month', target_topic]).iloc[:-1]

    # Create an XGBoost regressor
    model = xgb.XGBRegressor()
    # Fit the model on the training data
    model.fit(X_train, y_train)
    #model.save_model("year_month.json")

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Create a DataFrame with YearMonth and the predicted values
    y_pred_df = pd.DataFrame({'year_month': test_data['year_month'].iloc[:-1], 'Predicted': y_pred})

    # Merge the predicted DataFrame with the original test_data DataFrame
    merged_data = pd.merge(y_pred_df, test_data, on='year_month')

    # Calculate the root mean squared error (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot the predicted vs. actual values along with y_train
    fig=plt.figure(figsize=(12, 6))
    plt.plot(merged_data['year_month'], merged_data['Predicted'], label='Predicted', marker='x')
    plt.plot(merged_data['year_month'], merged_data[target_topic], label='Actual', marker='o')
    plt.plot(train_data['year_month'].iloc[:-1], y_train, label='Train', linestyle='--', color='gray')
    plt.xlabel('Time')
    plt.ylabel('Monthly publication frequency')
    plt.title(f'Topic {target_topic} Publication Frequency Forecast (RMSE: {rmse:.2f})')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)
# Create a select box for the user to choose a topic number
selected_topic = st.selectbox("Select a Topic Number", sorted(topic_mapping.values()))

# Create a button to trigger the prediction
if st.button("Predict"):
    # Get the selected topic code based on the user's selection
    
    selected_topic_code = [k for k, v in topic_mapping.items() if v == selected_topic][0]
    plot_bertopic_data_frequency(selected_topic_code, topic_mapping)