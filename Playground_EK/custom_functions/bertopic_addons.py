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
from umap import UMAP
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

## new column name
title_plus_abstract = 'title_abstract'
columns_to_keep = ['paperId', 'title', 'abstract', 'title_abstract', 'year', 'referenceCount', 'citationCount', 'influentialCitationCount']



## ========    DATA PULL    =======================================================



## LOCAL DATA PULL 
def pull_local_sql_database():
    
    print("Step 1/7: Loading the dataset ...")
    DB_STRING = os.getenv('DB_STRING')
    # Defining the Engine
    engine = sqlalchemy.create_engine(DB_STRING)

    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    
    ##### ========================================= REMOVE FOR LARGER ANALYSIS ============================
    #table_names = table_names[3:10]
    ##### ========================================= REMOVE FOR LARGER ANALYSIS ============================

    all_dataframes = pd.DataFrame()
    for index, sql_table_name in enumerate(table_names):
        with engine.connect() as conn, conn.begin():  
            df = pd.read_sql_table(sql_table_name, conn) 
            all_dataframes=pd.concat([all_dataframes, df])
            
    engine.dispose()
            
    return all_dataframes

## AWS DATA PULL 
def pull_aws_sql_database(sql_table_name):
    
    print("Step 1/7: Loading the dataset ...")
    DB_STRING = os.getenv('DB_STRING_AWS')
    # Defining the Engine
    engine = sqlalchemy.create_engine(DB_STRING)
    
    with engine.connect() as conn, conn.begin():  
        aws_dataframe = pd.read_sql_table(sql_table_name, conn) 
            
    return aws_dataframe




## ========    DATA PUSH TO AWS    =======================================================





def localSQL_to_awsSQL(table_name):
    ## pulls and combines everything in the local database into one dataframe
    raw_combined_dfs = pull_local_sql_database()
    ## data is cleaned 
    combined_cleaned_dataframe = cleaning_data_from_sql(raw_combined_dfs)
    ## 
    dataframe_to_aws_sql(combined_cleaned_dataframe, table_name)
    ## return combined dataframe
    return combined_cleaned_dataframe
    

## i need a function that then pushes back out to the cloud database
# input to function is table name
## code to connect to AWS 
def dataframe_to_aws_sql(infun_df, table_name):
    DB_STRING_AWS = os.getenv('DB_STRING_AWS')
    # Defining the Engine
    engine_aws = sqlalchemy.create_engine(DB_STRING_AWS)
    engine_aws.connect()
    with engine_aws.connect() as conn, conn.begin():  
        infun_df.to_sql(table_name, conn, if_exists='replace', index=False)
        



## ========    DATA CLEAN    =======================================================



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
    print("Step 5/7: We longer drop columns, its not needed ...")
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




## ========    BERTOPIC EXTRACTION    =======================================================



## extract topics
def extract_topics_with_bertopic(dataframe, min_topic_size, fine_tune_label, ngram):
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
    


## ========    BERTOPIC VISUALIZATION ADDITIONS    =======================================================

def merge_embeddings_to_df(infunc_df,infun_model):
    ## embeddings ====== calculates many vectors that then get reduced 
    embeddings = infun_model._extract_embeddings(infunc_df['title_abstract'].to_list(), method="document")

    ### embeddings in 3D space 
    reduced_embeddings = reduce_to_3d(embeddings)

    ### dataframe of embeddings in 3D space
    reduced_embeddings_dataframe= pd.DataFrame(reduced_embeddings)
    # rename column names
    renamed_embeddings_dataframe = reduced_embeddings_dataframe.rename(columns={0: 'x_vector', 1: 'y_vector', 2: 'z_vector'})
    ## merge vectors into dataframe
    vectorized_og_df = pd.concat([infunc_df, renamed_embeddings_dataframe], axis=1)
    return vectorized_og_df

def reduce_to_3d(embeddings):
    reducer = UMAP(n_components=3)  # You can adjust n_components as needed
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings



# ============================ TOPIC + PROBABILITY DATAFRAME 


def dataframing_topic_proba(infunch_model):
    topics_and_probabilities = infunch_model.get_topics()
    topic_code_expansion = list(topics_and_probabilities.keys())*10
    topic_code_expansion.sort()
    flat_list = list()
    for sub_list in topics_and_probabilities.values():
        flat_list += sub_list
    topics1=[]
    probabilities=[]
    for x in flat_list:
        topics1.append(x[0])
        probabilities.append(x[1])
        
    topic_proba_dict = {'topic_code': topic_code_expansion, 'topic': topics1, 'probability':probabilities}
    topic_prob_dict = pd.DataFrame(topic_proba_dict)
    return topic_prob_dict


## ================================= BERTOPIC STUFF 


dictionary_for_saved_data = {'aws_raw_sql_name':[],
                             'aws_vectorized_sql_name': [],
                             'aws_topic_proba_sql_name':[],
                             'model_name_filepath':[]
                             }


# def = pull_local_aws_database(sql_table_name): 
# ====== so give a name
def run_bertopic_model_push_SQLdata(aws_raw_name, aws_vectorized_name, aws_topic_proba_name, model_name):
    raw_aws_dfs = pull_aws_sql_database(aws_raw_name)
    ## cleans data
    cleaned_local_df = cleaning_data_from_sql(raw_aws_dfs)
    
    #### =============================    VARIABLES 
    min_topic_size = 50
    fine_tune_label = False
    ngram = 3
    
    #### =============================    VARIABLES 
    print("Step 1/4: RUNNING BERTOPIC ...")
    topified_df,model,topics = extract_topics_with_bertopic(cleaned_local_df, min_topic_size, fine_tune_label, ngram)
    
    ## save nmodel
    print("Step 2/4: SAVING MODEL ...")
    model.save("./nonSQL_database/" + model_name, serialization="pickle")
    
    # 2. VECTORS & TOPIFIED DATA
    print("Step 3/4: CREATING VECTORIZED TABLE ...")
    vectorized_df = merge_embeddings_to_df(topified_df,model)
    ## push table to AWS
    dataframe_to_aws_sql(vectorized_df, aws_vectorized_name)
    # def = pull_local_aws_database(sql_table_name): 
    
    ## dataframe _ of topics and probabilities
    print("Step 3/4: CREATING TOPIC PROBA ...")
    topic_proba_df = dataframing_topic_proba(model)
    ## push table to AWS
    dataframe_to_aws_sql(topic_proba_df, aws_topic_proba_name)
    
    dictionary_for_saved_data['aws_raw_sql_name'].append(aws_raw_name)
    dictionary_for_saved_data['aws_vectorized_sql_name'].append(aws_vectorized_name)
    dictionary_for_saved_data['aws_topic_proba_sql_name'].append(aws_topic_proba_name)
    dictionary_for_saved_data['model_name_filepath'].append(model_name)
    
    return dictionary_for_saved_data
