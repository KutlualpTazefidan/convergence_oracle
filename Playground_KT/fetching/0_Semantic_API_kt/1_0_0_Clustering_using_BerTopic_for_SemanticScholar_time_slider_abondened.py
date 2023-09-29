import pandas as pd
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_colwidth', 100)
from custom_functions import cf_clustering as cfc

# Load and the data set and apply a year filter:
years = [1900]
sy_df = cfc.load_clean_dataset("./Playground_KT/fetching/0_Semantic_API_kt/data/1880-2023_Science_48966.csv",years)
cfc.viz_study_field_distribution(sy_df)

# Create clusters for the selected single year: ...
sy_model,sy_topics,sy_proba = cfc.cluster_data(sy_df,min_topic_size=2,fine_tune_label=True,get_topics=True,get_proba=True,ngram=3)
# Saving the BERTopic model
sy_model.save("./Playground_KT/fetching/0_Semantic_API_kt/models/bertopic_model_"+str(years[0]))
sy_model.visualize_topics()

sy_embeddings = sy_model._extract_embeddings(sy_df['title_abstract_studyfield'].to_list(), method="document")

# 3d Distance for the topics
cfc.visualize_intertopic_distances_3d(sy_df,sy_topics,sy_model,sy_embeddings)
