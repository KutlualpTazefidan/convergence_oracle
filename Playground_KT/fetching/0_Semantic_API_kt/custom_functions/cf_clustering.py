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


# Function to load and clean the dataset
def load_clean_dataset(path,years=None):
    # Load the dataset from the specified path
    df = pd.read_csv(path)
    if years: df = df[df['year'].isin(years)]

    # Fill NaN values with empty strings
    df.fillna('', inplace=True)
    
    # List of columns to drop from the DataFrame
    columns_to_drop = [
        'paperId', 'paperId', 's2FieldsOfStudy', 'MAG', 'DOI', 'CorpusId',
        'ArXiv', 'DBLP', 'PubMed', 'PubMedCentral', 'id', 'issn',
        'alternate_issns', 'alternate_urls', 'type', 'publicationTypes', 'url'
    ]
    
    # Drop the specified columns from the DataFrame
    df_c = df.drop(columns=columns_to_drop, axis=1)


    def clean_text(row):
        # Combine 'title' and 'abstract' into a single column
        combined_text = row['title'] + ' ' + row['abstract']

        # Check if 'fieldsOfStudy' exists and is not empty
        if row['fieldsOfStudy']:
            # Parse the string representation of the list into an actual list
            fields_list = ast.literal_eval(row['fieldsOfStudy'])
            # Remove single quotes from each field of study
            cleaned_fields = [field.strip(" '") for field in fields_list]
            # Join the cleaned fields with commas and spaces
            field_of_study_str = ' '.join(cleaned_fields)
            combined_text += ' ' + field_of_study_str

        # Tokenize and remove stopwords
        cleaned_tokens = [word for word in combined_text.split() if word.lower() not in ENGLISH_STOP_WORDS]
        return ' '.join(cleaned_tokens)


    # Apply the clean_text function to each row to create a new 'title_abstract' column
    df_c['title_abstract_studyfield'] = df_c.apply(clean_text, axis=1)
    
    # Shuffle the DataFrame for randomness and reset the index
    df_c = df_c.sample(frac=1).reset_index(drop=True)
    
    print("Length of the dataset: ", len(df_c))
    # print("Null in the dataset",df.isnull().sum())
    # Return the cleaned DataFrame
    return df_c

# Function to cluster data using BERTopic
def cluster_data(df,min_topic_size=50,fine_tune_label=False,get_topics=False,get_proba=False,ngram=2):  
    # Extract documents from the 'title_abstract_studyfield' column
    docs = df['title_abstract_studyfield'].tolist()
    
    # Measure the start time for model fitting
    start_time = time.time()
    
    # Create and fit a BERTopic model
    if fine_tune_label:
        representation_model = KeyBERTInspired()
        model = BERTopic(language="english", n_gram_range=(1, ngram), min_topic_size=min_topic_size,representation_model=representation_model)
    else:
        model = BERTopic(language="english", n_gram_range=(1, ngram), min_topic_size=min_topic_size)
    
    topics, proba = model.fit_transform(docs)
    
    # Calculate the elapsed time for model fitting
    elapsed_time_minutes = (time.time() - start_time) / 60
    
    # Print a message with the elapsed time
    print(f"Model fitting completed in {elapsed_time_minutes:.2f} minutes")
    
    # Return the model
    if get_topics and get_proba:
        return model,topics,proba
    elif get_topics and not get_proba:
        return model,topics
    elif get_proba and not get_topics:
        return model,proba
    return model

# Function to visualize the distribution of study fields
def viz_study_field_distribution(df, vis_length=None):
    # Set the default visualization length to the length of the DataFrame if not provided
    if vis_length is None:
        vis_length = len(df)

    # Calculate the counts of study fields
    source_counts = df['fieldsOfStudy'].value_counts()[:vis_length]

    # Create a figure and a single axis for the bar chart
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot the study field counts as a bar chart
    source_counts.plot(kind='bar', color='lightblue', edgecolor='black', ax=ax)

    # Set the title and labels for the plot
    ax.set_title('Distribution of Study Fields')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(source_counts)))
    ax.set_xticklabels(source_counts.index, rotation=90, ha="right")

    # Ensure the plot layout is tight
    plt.tight_layout()

    # Display the plot
    plt.show()
    
# Expanding the df to include the topic labels
def add_topic_labels(df,topics,proba,model):
    df['topic_code'] = topics
    df['proba'] = proba
    topic_dict = model.get_topics()
    df['topic_list'] = df['topic_code'].apply(lambda x: topic_dict.get(x, []))
    df['best_topic'] = df['topic_code'].apply(lambda x: topic_dict.get(x, [])[0][0])
    display(df.head())
    return df

def visualize_intertopic_distances_3d(filename_prefix,df,topics,model,umap_embeddings):
    
    # Step 1: Reduce Embeddings to 3D using UMAP
    def reduce_to_3d(embeddings):
        reducer = UMAP(n_components=3)  # You can adjust n_components as needed
        umap_embeddings = reducer.fit_transform(embeddings)
        return umap_embeddings
        
    # Step 2: Calculate Centroids in 3D Space
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

    # Step 3: Visualization of Intertopic Distances in 3D
    def plot_intertopic_distances(centroids, topics,model):
        
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
        # fig.update_layout(legend_orientation="h")
        # Customize the layout and axis labels
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
        pyo.plot(fig,filename="./plots/3d_distance_map_for_"+str(filename_prefix)+"_"+str(len(topics))+"_topics.html",auto_open=False)
        fig.show()
        
    # # Customize the color scale if needed
    # color_scale = px.colors.sequential.Plasma  # Replace with a desired color scale
    # color_variable = topic_counts  # Replace 'petal_length' with the variable you want to use
    # fig.update_traces(marker=dict(colorbar=dict(title='Color Bar Title', tickvals=[min(color_variable), max(color_variable)], ticktext=['Min', 'Max']), colorscale=color_scale))

    # Assuming you have your embeddings, topics, and model
    print("Step 1/3: Reducing the dimensions of the embeddings ...")
    umap_embeddings = reduce_to_3d(umap_embeddings)  # Step 1
    print("Step 2/3: Calculating centroids ...")
    centroids = calculate_centroids(umap_embeddings, topics)  # Step 2
    print("Step 3/3: Visualizing intertopic distances ...")
    plot_intertopic_distances(centroids, topics,model)  # Step 3