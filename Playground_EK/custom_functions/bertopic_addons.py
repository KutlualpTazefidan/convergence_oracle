import pandas as pd
import ast
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time
from umap import UMAP
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import euclidean
from bertopic.representation import KeyBERTInspired
import plotly.graph_objs as go
# This package allows me to save the plots into html file
import plotly.offline as pyo
from plotly.express.colors import sample_colorscale
from IPython.display import display
from sqlalchemy import create_engine, inspect, text
import requests
import json
import pandas as pd
import time
import string
import json
import psycopg2
import os
import sqlalchemy
from dotenv import load_dotenv
load_dotenv()

## new column name
title_plus_abstract = 'title_abstract'
columns_to_keep = ['paperId', 'title', 'abstract', 'title_abstract', 'year', 'referenceCount', 'citationCount', 'influentialCitationCount']

def pull_sql_database():
    
    print("Step 1/7: Loading the dataset ...")
    DB_STRING = os.getenv('DB_STRING')
    # Defining the Engine
    engine = sqlalchemy.create_engine(DB_STRING)

    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    
    ##### ========================================= REMOVE FOR LARGER ANALYSIS ============================
    table_names = table_names[3:10]
    ##### ========================================= REMOVE FOR LARGER ANALYSIS ============================

    all_dataframes = pd.DataFrame()
    for index, sql_table_name in enumerate(table_names):
        with engine.connect() as conn, conn.begin():  
            df = pd.read_sql_table(sql_table_name, conn) 
            all_dataframes=pd.concat([all_dataframes, df])
            
    return all_dataframes

### clean data first, then extract topics
def cleaning_data_from_sql(dataframe):
    ## clean up dataset 
    # step 1: delete duplicates
    # step 2: deletes rows with empty abstracts
    print("Step 2/7: Cleaning the dataset ...")
    df = clean_dataset(dataframe)
        
    ## create column with title + abstract 
    print("Step 3/7: Creating the analysis column ...")
    df[title_plus_abstract] = df['title'] + ' ' + df['abstract']
    
    # df = include_study_field(df) # For debugging
    print("Step 4/7: Removing stopwords ...")
    df = remove_stopwords_from_column(df,title_plus_abstract)
    
    ## drop the columns not needed for analysis
    print("Step 5/7: Dropping columns ...")
    # df = df[columns_to_keep]
    return df
    
    
def clean_dataset(df):
    
    # Shuffle the DataFrame for randomness and reset the index
    df = df.sample(frac=1).reset_index(drop=True)
    
    df_length = len(df)
    
    number_of_duplicates = df.duplicated().sum()
    print(f"Dropping {str(number_of_duplicates)} duplicates out of {len(df)} data points.")
    df = df.drop_duplicates()

    number_of_nans = df['abstract'].isna().sum()
    print(f"Dropping {str(number_of_nans)} NaNs out of {len(df)} data points.")
    df.dropna(subset=['abstract'],inplace=True)

    print(f"Cleaned df size {len(df)}; Original df size: {df_length}")
    
    return df

# Remove stopwords from a text
def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in ENGLISH_STOP_WORDS]
    return " ".join(filtered_tokens)

## extract topics
def extract_topics_with_bertopic_eric(dataframe, min_topic_size, fine_tune_label, ngram):
    print("Step 6/7: Extracting topics ...")
    model, topics = extract_topics(dataframe, min_topic_size=min_topic_size, fine_tune_label=fine_tune_label, ngram=ngram)
    
    print("Step 7/7: Adding topic labels to the df ...")
    df = add_topic_labels(dataframe,topics,model)
    return df,model,topics



# Function to cluster data using BERTopic
def extract_topics(df,min_topic_size=50,fine_tune_label=False,ngram=2):  
    # Extract documents from the 'title_abstract_studyfield' column
    docs = df[title_plus_abstract].tolist()
    
    # Measure the start time for model fitting
    start_time = time.time()
    
    # Create and fit a BERTopic model
    representation_model = KeyBERTInspired() if fine_tune_label else None
    model = BERTopic(language="english", n_gram_range=(1, ngram), min_topic_size=min_topic_size,representation_model=representation_model)

    topics, proba = model.fit_transform(docs)
    
    # Calculate the elapsed time for model fitting
    elapsed_time_minutes = (time.time() - start_time) / 60
    
    # Print a message with the elapsed time
    print(f"Model fitting completed in {elapsed_time_minutes:.2f} minutes")
    
    # Return the model
    return model,topics

# Expanding the df to include the topic labels
def add_topic_labels(df,topics,model):
    df['topic_code'] = topics
    ## add one to topic code to match the index
    df['topic_code'] = df['topic_code'] + 1
    
    topic_dict = model.get_topic_info()['Name']
    df['topic_list'] = df['topic_code'].apply(lambda x: topic_dict.get(x, []))
    
    ## minus one to topic code to return to original value
    df['topic_code'] = df['topic_code']-1
    return df
     
def remove_stopwords_from_column(df, column_name):
    # Apply the remove_stopwords function to the specified column
    df[column_name] = df[column_name].apply(remove_stopwords)
    return df
    
    