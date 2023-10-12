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
import pandas as pd
from dotenv import load_dotenv
load_dotenv()





## AWS DATA PULL 
def pull_aws_sql_database(sql_table_name):
    
    print("Step 1/7: Loading the dataset ...")
    DB_STRING = os.getenv('DB_STRING_AWS')
    # Defining the Engine
    engine = sqlalchemy.create_engine(DB_STRING)
    
    with engine.connect() as conn, conn.begin():  
        aws_dataframe = pd.read_sql_table(sql_table_name, conn) 
            
    return aws_dataframe

## AWS DATA PULL 
def pull_local_sql_database(sql_table_name):
    
    print("Step 1/7: Loading the dataset ...")
    DB_STRING = os.getenv('DB_STRING_final')
    # Defining the Engine
    engine = sqlalchemy.create_engine(DB_STRING)
    
    with engine.connect() as conn, conn.begin():  
        sql_dataframe = pd.read_sql_table(sql_table_name, conn) 
            
    return sql_dataframe

## i need a function that then pushes back out to the cloud database
# input to function is table name
## code to connect to AWS 
def dataframe_to_final_local_sql(infun_df, table_name):
    DB_STRING = os.getenv('DB_STRING_final')
    # Defining the Engine
    engine = sqlalchemy.create_engine(DB_STRING)
    engine.connect()
    with engine.connect() as conn, conn.begin():  
        infun_df.to_sql(table_name, conn, if_exists='replace', index=False)