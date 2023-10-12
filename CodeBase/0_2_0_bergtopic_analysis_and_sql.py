# Nature
aws_raw_data_table = 'Nature_1869_2023'
aws_vectorized_data_table = 'topified_vectorized_Nature_1869_2023'
aws_topic_proba_data_table = 'topic_proba_data_Nature_1869_2023'
model_name = 'model_Nature_1869_2023'

## run bertopic analysis 
## run bertopic visualization data
from custom_functions import bertopic_addons as cfc

aws_raw_name = aws_raw_data_table
aws_vectorized_name = aws_vectorized_data_table
aws_topic_proba_name = aws_topic_proba_data_table
raw_aws_dfs = cfc.pull_aws_sql_database(aws_raw_name)
## cleans data
cleaned_local_df = cfc.cleaning_data_from_sql(raw_aws_dfs)

#### =============================    VARIABLES 
min_topic_size = 50
fine_tune_label = False
ngram = 3

#### =============================    VARIABLES 
print("Step 1/4: RUNNING BERTOPIC ...")
topified_df,model,topics = cfc.extract_topics_with_bertopic(cleaned_local_df, min_topic_size, fine_tune_label, ngram)

## save nmodel
print("Step 2/4: SAVING MODEL ...")
model.save("./CodeBase/nonSQL_database/" + model_name, serialization="pickle")