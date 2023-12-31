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

def load_clean_dataset(path,years=None):
    # Load the dataset from the specified path
    # check for duplicates/remove drop_duplicated
    # df.duplicated().value_counts()
    # df.drop_duplicates()
    # isna() values, 
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
        # if row['fieldsOfStudy']:
        #     # Parse the string representation of the list into an actual list
        #     fields_list = ast.literal_eval(row['fieldsOfStudy'])
        #     # Remove single quotes from each field of study
        #     cleaned_fields = [field.strip(" '") for field in fields_list]
        #     # Join the cleaned fields with commas and spaces
        #     field_of_study_str = ' '.join(cleaned_fields)
        #     combined_text += ' ' + field_of_study_str

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
    
    
def visualize_intertopic_distances_3d_time_dependent(filename_prefix,df,topics,model,umap_embeddings,include_unclassified=False):
    
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
    def plot_intertopic_distances(centroids, topics,model,include_unclassified):
            
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
                    size=sphere_radius[i], 
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
    plot_intertopic_distances(centroids, topics,model,include_unclassified)  # Step 3