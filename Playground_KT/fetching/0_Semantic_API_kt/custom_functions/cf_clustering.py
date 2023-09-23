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
        combined_text = row['title'] + '. ' + row['abstract']

        # Check if 'fieldsOfStudy' exists and is not empty
        if row['fieldsOfStudy']:
            # Parse the string representation of the list into an actual list
            fields_list = ast.literal_eval(row['fieldsOfStudy'])
            # Remove single quotes from each field of study
            cleaned_fields = [field.strip(" '") for field in fields_list]
            # Join the cleaned fields with commas and spaces
            field_of_study_str = ', '.join(cleaned_fields)
            combined_text += '. ' + field_of_study_str

        # Tokenize and remove stopwords
        cleaned_tokens = [word for word in combined_text.split() if word.lower() not in ENGLISH_STOP_WORDS]
        return ' '.join(cleaned_tokens)


    # Apply the clean_text function to each row to create a new 'title_abstract' column
    df_c['title_abstract_studyfield'] = df_c.apply(clean_text, axis=1)
    
    # Shuffle the DataFrame for randomness and reset the index
    df_c = df_c.sample(frac=1).reset_index(drop=True)
    
    print("Length of the dataset: ", len(df_c))
    print("Null in the dataset",df.isnull().sum())
    # Return the cleaned DataFrame
    return df_c

# Function to cluster data using BERTopic
def cluster_data(df,min_topic_size=50,fine_tune_label=False,get_topics=False,get_proba=False):  
    # Extract documents from the 'title_abstract_studyfield' column
    docs = df['title_abstract_studyfield'].tolist()
    
    # Measure the start time for model fitting
    start_time = time.time()
    
    # Create and fit a BERTopic model
    if fine_tune_label:
        representation_model = KeyBERTInspired()
        model = BERTopic(language="english", n_gram_range=(1, 2), min_topic_size=min_topic_size,representation_model=representation_model)
    else:
        model = BERTopic(language="english", n_gram_range=(1, 2), min_topic_size=min_topic_size)
    
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

def visualize_intertopic_distances_3d(df,topics,model,umap_embeddings):
    
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
        
        # Create a scatter plot with points for each topic centroid
        fig = go.Figure()
        for i, centroid in enumerate(centroids):
            # Calculate the sphere radius based on topic count
            sphere_radius = np.cbrt(topic_counts[i])
            
            # Create a sphere marker for each topic
            fig.add_trace(go.Scatter3d(
                x=[centroid[0]],
                y=[centroid[1]],
                z=[centroid[2]],
                mode='markers',
                marker=dict(size=sphere_radius, color='blue', opacity=0.5),
                text=f'Topic {best_labels[i]}, Count: {topic_counts[i]}'
            ))
            
        # Customize the color scale if needed
        color_scale = px.colors.sequential.Plasma  # Replace with a desired color scale
        color_variable = topic_counts  # Replace 'petal_length' with the variable you want to use
        fig.update_traces(marker=dict(colorbar=dict(title='Color Bar Title', tickvals=[min(color_variable), max(color_variable)], ticktext=['Min', 'Max']), colorscale=color_scale))

        # Customize the layout if needed
        fig.update_layout(scene=dict(aspectmode='cube'))
        # Customize the layout and axis labels
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        
        # Show the 3D scatter plot
        fig.show()
        
    # Assuming you have your embeddings, topics, and model
    print("Step 1/3: Reducing the dimensions of the embeddings ...")
    umap_embeddings = reduce_to_3d(umap_embeddings)  # Step 1
    print("Step 2/3: Calculating centroids ...")
    centroids = calculate_centroids(umap_embeddings, topics)  # Step 2
    print("Step 3/3: Visualizing intertopic distances ...")
    plot_intertopic_distances(centroids, topics,model)  # Step 3