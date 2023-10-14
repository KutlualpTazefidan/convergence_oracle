
### Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
import nltk
from collections import Counter
from wordcloud import WordCloud
import plotly.io as pio
import plotly.graph_objects as go

## Data exploration

### Task1:
### Load the 'penguins_pimped.csv' file into a data frame df
### (it is under the folder data/ )
### Print out 5 random sample from df 
### (Hint: apply the function sample() on df)

st.header('Convergence Oracle')
st.image('output.png')
# st.write('wrire whatever you want')


df = pd.read_csv('C:/Users/sa3id/spiced/test/convergence_oracle/Playground_SB/Topified_vectorized_Science1900_2023/web/data/topified_vectorized_Science_1900_2023_cleaned_up.csv')
df_sample = df.sample(5)

st.header('Science Data')
st.markdown('''Science Data 
1. Publication Year Distribution: - The dataset contains publications spanning a wide range of years, with a noticeable peak in the 2000s. - This suggests a significant volume of research during that decade.
2. Correlation Analysis: - There is a strong positive correlation between citation Count and influential Citation Count, indicating that highly cited papers tend to have a greater impact.
3. Field of Study Analysis: - The most common field of study in the dataset is Medicine, suggesting that a substantial portion of the publications belongs to this field. - Other fields of study, while present, may have lower representation in the dataset.
4. Publication Type Distribution: - The majority of publications in the dataset are of type Journal Article, indicating that traditional journal articles dominate the dataset.
5. Additional Insights:  explore trends and patterns within specific fields of study or publication types. - Text analysis of titles and abstracts to reveal common themes or topics within the publications.''')

st.dataframe(df_sample)

Topic = df['topic_list'].unique()
### 2.2 Display the data for an island you choose from the dataframe 
my_Topic = df['topic_list']
my_Topic_df =df[df['topic_list'] == my_Topic]
#to create sub-section
st.subheader('Select a topic')
user_topic= st.selectbox(label='Select a topic',options=my_Topic)

if st.checkbox('Filier the data'):
    st.dataframe(df[df['topic_list'] == user_topic])
### Plotting
st.markdown('---')

######################  Create the scatter plot
st.markdown('---')
st.header('Vector Data Analysis')

fig = px.scatter_3d(df, x='x_vector', y='y_vector', z='z_vector',
                     title='3D Scatter Plot for Vectors',
                     color='topic_list',
                     labels={'x_vector': 'X Vector', 'y_vector': 'Y Vector', 'z_vector': 'Z Vector'},
                     template='plotly_dark',
                     )  # Add animation_frame to enable rotation

fig.update_traces(marker=dict(size=3, opacity=0.5), selector=dict(mode='markers'))

fig.update_layout(
    scene=dict(
        xaxis_title='X Vector',
        yaxis_title='Y Vector',
        zaxis_title='Z Vector',
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12,
        zaxis_tickfont_size=12,
    ),
    font=dict(size=1),
    margin=dict(l=0, r=0, b=0, t=40),
    height=800,
    width=1200,
)

st.plotly_chart(fig)
st.markdown('---')




######################  Topics over time
st.header('Topics over time')

# Convert the year column to a numeric format
df['year'] = pd.to_datetime(df['year']).dt.year

# Sort the DataFrame by year in ascending order
df = df.sort_values(by='year', ascending=True)

# Create a select box to filter data by topic with "All Topics" as the default choice
selected_topic = st.selectbox('Select a topic to filter the data', ['All Topics'] + list(df['topic_list'].unique()))

# Filter the data based on the selected topic
if selected_topic != 'All Topics':
    filtered_df = df[df['topic_list'] == selected_topic]
else:
    filtered_df = df  # Show all topics

# Create the scatter plot
fig2 = px.scatter(data_frame=filtered_df[filtered_df["year"] >= 1900],
                  x="topic_code",
                  y="citationCount",
                  animation_frame="year",
                  color="z_vector",
                  hover_name="topic_list",
                  size="citationCount",
                  size_max=450)

# Update the y-axis range
fig2.update_yaxes(range=[0, 400])

# Show the updated figure within the Streamlit app
st.plotly_chart(fig2)



# # plotting a source
st.markdown('---')
st.header('Sources')
st.write('wrire whatever you want')

st.markdown('---')
st.header('Data')
st.write('AWS-tables')