from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from xgboost import plot_importance
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

def cumsum_or_moving_average(infun_df, cumlative, infun_moving_ave):
    # convert target topic to a moving average
    list_of_topic_codes = list(infun_df.columns.drop('date'))
    if cumlative == True:
        ## now predicting the moving average
        infun_df[list_of_topic_codes] = infun_df[list_of_topic_codes].cumsum()
    else:    
        ## now predicting the moving average
        infun_df[list_of_topic_codes] = infun_df[list_of_topic_codes].rolling(infun_moving_ave).mean()
    return infun_df

def clean_and_pivot_dataset(infun_df, infun_type_of_metric, infun_timeframe):
    
## you can change this to all topics 
    aws1_dfs= infun_df
    aws1_dfs['topic_code'] = pd.to_numeric(aws1_dfs['topic_code'], downcast='integer')
    aws1_dfs[infun_type_of_metric] = pd.to_numeric(aws1_dfs[infun_type_of_metric], downcast='integer')
    aws1_dfs = aws1_dfs[aws1_dfs['publicationDate'] != 'None'].reset_index()
    aws1_dfs['publicationDate'] = pd.to_datetime(aws1_dfs['publicationDate'], format='%Y-%m-%d')
    aws1_dfs['date'] = aws1_dfs['publicationDate'].dt.to_period(infun_timeframe)
    aws1_dfs['date']=aws1_dfs['date'].astype(str)
    aws1_dfs['date']=pd.to_datetime(aws1_dfs['date'])
    # ==== LEGACY NO NEED :)) aws1_dfs= aws1_dfs[aws1_dfs['topic_code']!=-1]

    ## code to change the date time, for the moment use year
    grouped_df = aws1_dfs.groupby(['topic_code', 'date'])[infun_type_of_metric].sum().reset_index()

    df = grouped_df.pivot(index='date', columns='topic_code', values=infun_type_of_metric)

    df.fillna(0, inplace=True)
    df = df.reset_index()
    return df


def plot_bertopic_data_into_future(dataframe, topic_number, future_years, moving_average, type_of_metric, timeframe, cumlative):
    
    ##### ============================ CLEANING UP DATAFRAME 
    # ## number of years into the future you can forecast
    if timeframe == 'M':
        n = future_years*12
        moving_average = moving_average*12
    else:
        n = future_years
        
    # Define the target column (Topic you want to forecast)
    target_topic = topic_number  # Change this to the Topic you want to forecast
    
    
    
    ## pivot df and clean the dataset
    pivot_df = clean_and_pivot_dataset(dataframe, type_of_metric, timeframe)
    # convert target topic to a moving average or cumalative 
    pivot_df = cumsum_or_moving_average(pivot_df, cumlative=cumlative, infun_moving_ave=moving_average)
    
    
    
    
    ## ================================  forecasting rearrangement

    # Dropping last n rows using drop
    target_column = pivot_df[target_topic]
    
    # forget about the lost values
    ## dropped the oldest columns
    target_column.drop(target_column.head(n).index, inplace = True)

    ## pivot_df['year'] + pd.offsets.DateOffset(years=5)
    untarget_columns = pivot_df.drop(target_topic, axis=1)

    ## keep this for later -- these are your forecasting columns 
    X_forecasting_data = untarget_columns.tail(n)
    X_forecasting_years = untarget_columns['date'].tail(n) + pd.offsets.DateOffset(years=future_years)
    X_forecasting_data['date'] = X_forecasting_years
    X_forecasting = X_forecasting_data.drop(columns=['date'])

    ##  drop the columns you keep for forecasting
    untarget_columns.drop(target_column.tail(n).index, inplace = True)

    df = pd.concat([untarget_columns, target_column.reset_index(drop=True)], axis=1)

    ## shift the prediction of each year
    df['date'] = df['date'] + pd.offsets.DateOffset(years=future_years)

    pivot_df = df    
    
    
    
    ### ================== TEST TRAIN SPLIT 
    # Separating the training set and testing set
    train_data=pivot_df[pivot_df['date'].dt.year<2011].reset_index(drop = True)
    test_data=pivot_df[pivot_df['date'].dt.year>2010].reset_index(drop = True)

    # Prepare the training and testing data
    X_train = train_data.drop(target_topic, axis=1)
    X_test = test_data.drop(target_topic, axis=1)

    # Shift the target column to align with next year's frequency
    y_train = train_data[target_topic].shift(-1).dropna()
    y_test = test_data[target_topic].shift(-1).dropna()

    # Exclude the 'YearMonth' column from the training and testing data
    X_train = train_data.drop(columns=['date', target_topic]).iloc[:-1]
    X_test = test_data.drop(columns=['date', target_topic]).iloc[:-1]
    
    
    
    
    ## ======================= MODEL 

    # Create an XGBoost regressor
    if cumlative == True: 
        model = Lasso()
    else:    
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
    

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    y_forecasting = model.predict(X_forecasting)

    # Create a DataFrame with YearMonth and the predicted values
    y_pred_df = pd.DataFrame({'date': test_data['date'].iloc[:-1], 'Predicted': y_pred})
    y_fore_df = pd.DataFrame({'date': X_forecasting_years, 'Forecasted': y_forecasting})

    # Merge the predicted DataFrame with the original test_data DataFrame
    merged_data = pd.merge(y_pred_df, test_data, on='date')
    merged_forecasted_data = pd.merge(y_fore_df, X_forecasting_data, on='date')

    #print(merged_forecasted_data)
    # Calculate the root mean squared error (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    #print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    
    
    ## =============================   BUILDING DATA TABLE 
    
    y_actual_test_df = test_data[['date',target_topic]]
    y_actual_train_df = train_data[['date',target_topic]]
    y_fore_df =y_fore_df.rename(columns={'Forecasted': target_topic})
    y_pred_df =y_pred_df.rename(columns={'Predicted': target_topic})

    y_actual_test_df['value'] = 'Actual_Test'
    y_actual_train_df['value'] = 'Actual_Train'
    y_fore_df['value'] = 'Forecasted'
    y_pred_df['value'] = 'Predicted_Test'

    final_data = pd.concat([y_actual_test_df,y_actual_train_df,y_fore_df,y_pred_df], axis=0)
    final_data =final_data.rename(columns={target_topic: 'quantity'})
    final_data['topic_code'] = target_topic
    final_data['RMSE'] = rmse
    final_data['metric'] = type_of_metric
    
    
    if cumlative == True: 
        final_data['cumsum_or_moving_average'] = 'cumsum'
    else: 
        final_data['cumsum_or_moving_average'] = 'moving_average'
    
    
    
    ### add moving average or cum sum
    
    return final_data


def tableau_tables(tab_df, tab_forecasting, tab_moving_ave, tab_cumsum):
    ## variables
    dataframe = tab_df
    forecasting_years = tab_forecasting
    moving_average_years = tab_moving_ave
    cumlative_boolean = tab_cumsum
    
    ## True goes with Monthly forecasting
    if cumlative_boolean == True: 
        timeframe = 'M'
    else: 
        timeframe = 'Y'
    
    ## 
    dataframe['count'] = 1

    ## create an ordered list of topic codes 
    topic_code_list = list(pd.Series(dataframe['topic_code'].sort_values()).unique())

    ## create a dataframe to concat later 
    growing_df = pd.DataFrame()

    ## for loop through the topic code list 
    for metric in ['count', 'citationCount', 'influentialCitationCount']:
        for topic_num in topic_code_list:
            print(topic_num)
            current_df = plot_bertopic_data_into_future(dataframe, topic_num, forecasting_years, moving_average_years, metric, timeframe, cumlative=cumlative_boolean)
            growing_df = pd.concat([current_df, growing_df]) 
        
    ## reset the index and drop the inxiex column 
    growing_df = growing_df.reset_index().drop('index',axis=1)

    ## match the topic code to the topic description 
    topic_codes = dataframe[['topic_code','topic_list']].drop_duplicates().reset_index().drop('index',axis=1)

    ## merge the topics descriptions to the data set
    full_dataset = growing_df.merge(topic_codes, on='topic_code')
    return full_dataset


def final_tableau_table(final_df, final_forecasting, final_moving_ave):
    cumsum_df = tableau_tables(tab_df= final_df, tab_forecasting= final_forecasting, tab_moving_ave= final_moving_ave, tab_cumsum= True)
    other_df = tableau_tables(tab_df= final_df, tab_forecasting= final_forecasting, tab_moving_ave= final_moving_ave, tab_cumsum= False)
    final_df = pd.concat([other_df, cumsum_df]) 
    return final_df
    