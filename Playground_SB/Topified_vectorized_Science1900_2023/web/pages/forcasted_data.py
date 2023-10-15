import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import io
import PIL
import base64
import streamlit as st
pivoted_data = pd.read_csv('C:/Users/sa3id/spiced/CS/convergence_oracle/Playground_SB/Topified_vectorized_Science1900_2023/web/pages/pivoted_data.csv')

st.header('Forcasted saved Data')
df = pivoted_data

st.title("Interactive Topic Selection")

selected_topic = st.selectbox("Select a Topic:", df['topic_list'].unique())

def update_plot(selected_topic):
    fig = go.Figure()

    topic_data = df[df['topic_list'] == selected_topic]

    # traces for Actual, Predicted, and Forecasted values
    fig.add_trace(go.Scatter(x=topic_data['date'], y=topic_data['Actual'], mode='lines', name=f'Actual - {selected_topic}'))
    fig.add_trace(go.Scatter(x=topic_data['date'], y=topic_data['Predicted'], mode='lines', name=f'Predicted - {selected_topic}'))
    fig.add_trace(go.Scatter(x=topic_data['date'], y=topic_data['Forecasted'], mode='lines', name=f'Forecasted - {selected_topic}'))

    # Use Matplotlib to add Train data
    plt.figure(figsize=(50, 24))
    plt.plot(topic_data['date'], topic_data['Train'], label=f'Train - {selected_topic}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Train Values Over Time for Topic: {selected_topic}')
    plt.legend()

    # Save the Matplotlib plot as an image and convert it to a data URI
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = PIL.Image.open(buf)
    img_data = io.BytesIO()
    img.save(img_data, format="PNG")
    img_uri = "data:image/png;base64," + base64.b64encode(img_data.getvalue()).decode()

    # Close the Matplotlib figure ########## this will save a lot of memory 
    plt.close()

    # Add the image to the Plotly figure using the data URI as the 'source' property
    fig.add_layout_image(
        x=0,
        y=0.25,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        source=img_uri,
        opacity=1,
        layer="below"
    )

    # Set layout properties
    fig.update_layout(
        title=f'Actual vs. Predicted vs. Forecasted for Topic: {selected_topic}',
        xaxis_title='Date',
        yaxis_title='Value',
        showlegend=True,
        xaxis_rangeslider_visible=True
    )

    # Show the Plotly figure
    st.plotly_chart(fig)

# Call the update_plot function with the initial selected_topic
update_plot(selected_topic)
st.markdown('---')
