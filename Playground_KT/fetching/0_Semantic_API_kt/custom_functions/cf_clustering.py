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


def extract_topics_with_bertopic(input_column_name, columns_to_drop, min_topic_size, fine_tune_label, ngram,years=None,filepath=None,folder_path=None):
    
    print("Step 1/7: Loading the dataset ...")
    if folder_path:
        df = combine_csv_files_to_df(folder_path)
    else:
        df = load_csv_file_to_df(filepath)
    if years: df = df[df['year'].isin(years)]
    print("Step 2/7: Cleaning the dataset ...")
    df = clean_dataset(df)
    print("Step 3/7: Creating the analysis column ...")
    df = combine_columns(df,"title","abstract",input_column_name)
    # df = include_study_field(df) # For debugging
    print("Step 4/7: Removing stopwords ...")
    df = remove_stopwords_from_column(df,input_column_name)
    print("Step 5/7: Dropping columns ...")
    df = drop_columns(df,columns_to_drop)
    print("Step 6/7: Extracting topics ...")
    model, topics, proba,docs = extract_topics(df,input_column_name, min_topic_size=min_topic_size, fine_tune_label=fine_tune_label, ngram=ngram)
    print("Step 7/7: Adding topic labels to the df ...")
    df = add_topic_labels(df,topics,proba,model)
    return df,model,topics,proba,docs

def visualize_bertopic_results(filename,df,topics,model,docs,input_column_name):
    print("Step 1/7: Creating barchart")
    fig_bar_chart = model.visualize_barchart(top_n_topics=10)
    print("Step 2/7: Creating 2d distance map")
    fig_2d_distance_map = model.visualize_topics()
    print("Step 3/7: Creating 2d distance map")
    fig_heatmap = model.visualize_heatmap()
    print("Step 4/7: Creating 2d distance map")
    fig_heatmap = model.visualize_heatmap()
    print("Step 5/7: Creating hierarchical topics")
    hierarchical_topics = model.hierarchical_topics(docs)
    fig_hierarchical_clustering = model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    print("Step 6/7: Creating visualize hierarchical documents map")
    embeddings = model._extract_embeddings(df[input_column_name].to_list(), method="document")
    # sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    # embeddings2 = sentence_model.encode(docs, show_progress_bar=False)
    # display("embeddings2",embeddings2)
    
    try:
        fig_hierarchy_in_doc_topics = model.visualize_hierarchical_documents(docs, hierarchical_topics, embeddings=embeddings)
        pyo.plot(fig_hierarchy_in_doc_topics, filename="./plots/"+filename+'_hierarchy_in_doc_topics_'+str(len(topics))+"_topics.html")
        display(fig_hierarchy_in_doc_topics)
    except Exception as e:
        # Handle the exception (e.g., print an error message)
        print(f"An error occurred during fig_hierarchy_in_doc_topics")
    pyo.plot(fig_2d_distance_map, filename="./plots/"+filename+'_2d_distance_map_'+str(len(topics))+"_topics.html")
    pyo.plot(fig_bar_chart, filename="./plots/"+filename+'_bar_chart_'+str(len(topics))+"_topics.html")
    pyo.plot(fig_heatmap, filename="./plots/"+filename+'_heatmap_'+str(len(topics))+"_topics.html")
    pyo.plot(fig_hierarchical_clustering, filename="./plots/"+filename+'_hierarchical_clustering_'+str(len(topics))+"_topics.html")
    display(fig_bar_chart)
    display(fig_2d_distance_map)
    display(fig_heatmap)
    display(fig_hierarchical_clustering)
    print("Step 7/7: Creating 3d intertopic distances map")
    visualize_intertopic_distances_3d(filename,df,topics,model,embeddings,include_unclassified=False)

# This function is for debugging / to show negative impact of study field if included
def include_study_field(df):
    def iterate_over_study_field(row):
        # Combine 'title' and 'abstract' into a single column
        combined_text = row['title_abstract']

        # Check if 'fieldsOfStudy' exists and is not empty
        if row['fieldsOfStudy']:
            # Parse the string representation of the list into an actual list
            fields_list = ast.literal_eval(row['fieldsOfStudy'])
            # Remove single quotes from each field of study
            cleaned_fields = [field.strip(" '") for field in fields_list]
            # Join the cleaned fields with commas and spaces
            field_of_study_str = ' '.join(cleaned_fields)
            combined_text += ' ' + field_of_study_str
        return combined_text
        
    # Apply the clean_text function to each row to create a new 'title_abstract' column
    df.fillna('', inplace=True)
    df['title_abstract'] = df.apply(iterate_over_study_field, axis=1)
    return df

def load_csv_file_to_df(file_path):
    df = pd.read_csv(file_path)
    return df

def combine_csv_files_to_df(folder_path):
    combined_df = pd.DataFrame()
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Construct the full path to the CSV file
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file into a DataFrame
            csv_df = load_csv_file_to_df(file_path)
            # Concatenate the current DataFrame with the combined DataFrame
            combined_df = pd.concat([combined_df, csv_df], ignore_index=True)
    return combined_df

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

def combine_columns(df:pd.DataFrame,column_name_1:str,column_name_2:str,new_column_name:str):
    if df[column_name_1].apply(lambda x: isinstance(x, list)).any():
        df[column_name_1].apply(lambda x: x[0])
    if df[column_name_2].apply(lambda x: isinstance(x, list)).any():
        df[column_name_2].apply(lambda x: x[0])
        
    df[new_column_name] = df[column_name_1] + df[column_name_2]
    return df

# Remove stopwords from a text
def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in ENGLISH_STOP_WORDS]
    return " ".join(filtered_tokens)

def remove_stopwords_from_column(df, column_name):
    # Apply the remove_stopwords function to the specified column
    df[column_name] = df[column_name].apply(remove_stopwords)
    return df

def drop_columns(df,columns_to_drop):
    if not columns_to_drop==[]:
        df = df.drop(columns=columns_to_drop, axis=1)
    return df

# Function to cluster data using BERTopic
def extract_topics(df,input_column_name,min_topic_size=50,fine_tune_label=False,ngram=2):  
    # Extract documents from the 'title_abstract_studyfield' column
    docs = df[input_column_name].tolist()
    
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
    return model,topics,proba,docs

def get_topic_info(topic_dict, topic_code, indices):
    result = topic_dict.get(topic_code, None)

    if result is not None:
        if isinstance(result, list) and len(result) > indices[0]:
            result = result[indices[0]][indices[1]]
        else:
            return ""

    return result

# Expanding the df to include the topic labels
def add_topic_labels(df,topics,proba,model):
    df['topic_code'] = topics
    df['proba'] = proba
    topic_dict = model.get_topics()
    df['topic_list'] = df['topic_code'].apply(lambda x: topic_dict.get(x, []))
    df['topic_1_label'] = df['topic_code'].apply(lambda x: get_topic_info(topic_dict, x, [0,0]))
    df['topic_1_proba'] = df['topic_code'].apply(lambda x: get_topic_info(topic_dict, x, [0,1]))
    df['topic_2_label'] = df['topic_code'].apply(lambda x: get_topic_info(topic_dict, x, [1,0]))
    df['topic_2_proba'] = df['topic_code'].apply(lambda x: get_topic_info(topic_dict, x, [1,1]))
    df['topic_3_label'] = df['topic_code'].apply(lambda x: get_topic_info(topic_dict, x, [2,0]))
    df['topic_3_proba'] = df['topic_code'].apply(lambda x: get_topic_info(topic_dict, x, [2,1]))
    return df

def visualize_intertopic_distances_3d(filename_prefix,df,topics,model,embeddings,include_unclassified=False):
        
    # Assuming you have your embeddings, topics, and model
    print("Step 1/3: Reducing the dimensions of the embeddings ...")
    umap_embeddings = reduce_to_3d(embeddings)  # Step 1
    print("Step 2/3: Calculating centroids ...")
    centroids = calculate_centroids(umap_embeddings, topics)  # Step 2
    print("Step 3/3: Visualizing intertopic distances ...")
    plot_intertopic_distances(centroids, topics,model,include_unclassified,filename_prefix)  # Step 3

# Reduce Embeddings to 3D using UMAP
def reduce_to_3d(embeddings):
    reducer = UMAP(n_components=3)  # You can adjust n_components as needed
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

# Calculate Centroids in 3D Space
def calculate_centroids(umap_embeddings, topics):
    unique_topics = np.unique(topics)
    centroids = np.zeros((len(unique_topics), 3))  # Initialize centroids array
    for i, topic in enumerate(unique_topics):
        # Find indices of documents belonging to the current topic
        topic_indices = np.where(topics == topic)[0]
        
        # Extract 3D coordinates for documents in the current topic
        topic_embeddings = umap_embeddings[topic_indices]
        
        # Calculate the centroid by averaging coordinates
        centroid = np.mean(topic_embeddings, axis=0)
        
        # Assign the centroid to the corresponding row in the centroids array
        centroids[i] = centroid

    return centroids
    
def plot_intertopic_distances(centroids, topics,model,include_unclassified,filename_prefix):
        
    topic_dict = model.get_topics()
    best_labels  = []
    for i in range(len(topic_dict)):
        best_labels.append(topic_dict[i-1][0][0])
        
    # Convert your 'topics' list to a NumPy array
    topics_shifted = np.array(topics)

    # Add 1 to all topic labels to include -1
    topics_shifted += 1

    # Now you can calculate topic counts using np.bincount
    topic_counts = np.bincount(topics_shifted)
            
    if not include_unclassified:
        if -1 in topic_dict:
            topic_counts = topic_counts[1:]
            centroids = centroids[1:]
            best_labels.pop(0)
    # print("topics: ",topics)
    # Calculate the sphere radius based on topic count
    sphere_radius = np.cbrt(topic_counts)
    
    # Calculate the sphere radius based on topic count
    max_radius = np.max(sphere_radius)
    min_radius = np.min(sphere_radius)
    normalized_radius = (sphere_radius - min_radius) / (max_radius - min_radius)
    
    rgb_colors = sample_colorscale('Viridis_r', list(normalized_radius))
    
    # Create a scatter plot with points for each topic centroid
    fig = go.Figure()
    for i, centroid in enumerate(centroids):
        
        # Capitalize the first letter of the topic label
        capitalized_label = best_labels[i].capitalize()
        if i == len(normalized_radius)-1: 
            hideScale = True
        else:
            hideScale = False
            
        # Create a sphere marker for each topic
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]],
            y=[centroid[1]],
            z=[centroid[2]],
            mode='markers',
            marker=dict(
                size=sphere_radius[i]*5, 
                color= rgb_colors[i],
                colorscale='Viridis_r',
                opacity=0.6,
                colorbar=dict(
                    title="Citation Count",
                    yanchor="top",
                    y=1.1,
                    xanchor="left",
                    x=-0.065,
                    # ticks="outside",
                ),
                cmin=min(topic_counts),
                cmax=max(topic_counts),
                showscale=hideScale
            ),
            name=f'{capitalized_label}, Count: {topic_counts[i]}',
            text=f'Topic {capitalized_label}, Count: {topic_counts[i]}'
        ))

    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis_title='', 
            yaxis_title='', 
            zaxis_title='',
            xaxis=dict(
                showticklabels=False,  # Disable tick labels on the x-axis
            ),
            yaxis=dict(
                showticklabels=False,  # Disable tick labels on the y-axis
            ),
            zaxis=dict(
                showticklabels=False,  # Disable tick labels on the z-axis
            ),
            # camera=dict(
            #     eye=dict(x=0.1, y=0, z=1.5)  # Set the initial camera position
            # )
            # dragmode='orbit',  # Limit interaction to orbit mode
        ),
        coloraxis=dict(
            colorbar=dict(
                title="Citation Count",
                showticklabels=False,  # Hide colorbar ticks

            )
        ),
        title='Distance for Topic Clusters'  # Add your desired title here
    )
    
    # Show the 3D scatter plot
    pyo.plot(fig,filename="./plots/"+str(filename_prefix)+"_3d_distance_map_for_"+str(len(topics))+"_topics.html",auto_open=False)
    fig.show()
    
    