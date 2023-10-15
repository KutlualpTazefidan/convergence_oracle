import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
import tempfile

# Load data from the CSV file
df = pd.read_csv('C:/Users/sa3id/spiced/test/convergence_oracle/Playground_SB/Topified_vectorized_Science1900_2023/web/data/topified_vectorized_Science_1900_2023_cleaned_up.csv')

# Create the initial 3D scatter plot using the specified columns
fig = px.scatter_3d(df, x='x_vector', y='y_vector', z='z_vector',
                     title='3D Scatter Plot for Vectors',
                     color='topic_list',
                     labels={'x_vector': 'X Vector', 'y_vector': 'Y Vector', 'z_vector': 'Z Vector'},
                     template='plotly_dark'              
                     )

fig.update_traces(marker=dict(size=3, opacity=0.5), selector=dict(mode='markers'))

fig.update_layout(
    scene=dict(
        xaxis_title='X Vector',
        yaxis_title='Y Vector',
        zaxis_title='Z Vector',
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12,
        zaxis_tickfont_size=12,
        camera=dict(
            eye=dict(x=-1.25, y=2, z=0.5),
            center=dict(x=0, y=0, z=0)
        )
    ),
    font=dict(size=1),
    margin=dict(l=0, r=0, b=0, t=0),
    height=700,
    width=900,  

)

# Add rotation animation to the plot
def rotate_z(x, y, z, theta):
    w = x + 1j * y
    return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

frames = []

for t in np.arange(0, 6.26, 0.1):
    xe, ye, ze = rotate_z(-1.25, 2, 0.5, -t)
    frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))

fig.frames = frames

# Save the Plotly figure as an HTML file
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
    fig.write_html(tmpfile.name)

# Display the HTML file using an iframe
with open(tmpfile.name, 'r', encoding='utf-8') as html_file:
    st.write("3D Scatter Plot with rotation and zoom (scroll to zoom in/out and rotate):")
    st.components.v1.html(html_file.read(), width=1000, height=600)
