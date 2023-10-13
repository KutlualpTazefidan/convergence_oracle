import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Use TensorFlow 1.x

# ... (previous code) ...

# Load your data from a CSV file (replace 'your_data.csv' with your actual data file)
df = pd.read_csv('Playground_SB\law\data_with_topics.csv')

# Step 1: Export DataFrame to TSV
df.to_csv('embedding_data_projector.tsv', sep='\t', index=False)

# Step 2: Define Embeddings (Random Embeddings for Illustration)
# Replace this with your actual embeddings if you have them
embedding_dimension = 300  # Adjust this to match your embeddings dimension
num_data_points = len(df)
embeddings = np.random.randn(num_data_points, embedding_dimension)


# Step 3: Create a TensorFlow Variable to hold the embeddings
embedding_var = tf.Variable(embeddings, name='embedding')

# Create a TensorFlow session
sess = tf.InteractiveSession()
projector_directory = 'projector_data'

# Initialize variables
sess.run(tf.global_variables_initializer())
# Create a summary writer for TensorBoard
summary_writer = tf.summary.FileWriter(projector_directory)

# Add embeddings to the projector
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = 'metadata_projector.tsv'

# Save the projector config
projector.visualize_embeddings(summary_writer, config)

# Write embeddings to the event file
saver = tf.train.Saver()
saver.save(sess, os.path.join(projector_directory, 'model.ckpt'))

# Close the session and summary writer
sess.close()
summary_writer.close()



# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorboard.plugins import projector
# import os

# # Load your data from a CSV file (replace 'your_data.csv' with your actual data file)
# df = pd.read_csv('data_with_topics.csv')

# # Step 1: Export DataFrame to TSV
# df.to_csv('embedding_data_projector.tsv', sep='\t', index=False)

# # Step 2: Define Embeddings (Random Embeddings for Illustration)
# # Replace this with your actual embeddings if you have them
# embedding_dimension = 300  # Adjust this to match your embeddings dimension
# num_data_points = len(df)
# embeddings = np.random.randn(num_data_points, embedding_dimension)

# # Step 3: Create a TensorFlow Variable to hold the embeddings
# embedding_var = tf.Variable(embeddings, name='embedding')

# # Step 4: Projector Configuration
# config = projector.ProjectorConfig()

# # Specify the embeddings tensor name and metadata path
# embedding = config.embeddings.add()
# embedding.tensor_name = embedding_var.name
# embedding.metadata_path = 'metadata_projector.tsv'  # Metadata file for additional information

# # Step 5: Save Metadata (Assuming you have metadata for each data point)
# metadata = df[['title', 'year', 'citationCount', 'topic']]
# metadata.to_csv('metadata_projector.tsv', sep='\t', index=False)

# # Step 6: Create the directory for the projector files
# projector_directory = 'projector_data'
# os.makedirs(projector_directory, exist_ok=True)

# # Step 7: Save the projector config
# projector.visualize_embeddings(projector_directory, config)

# # Step 8: Launch TensorBoard to view the embeddings projector
# # Open a terminal and navigate to the directory where your code is located.
# # Run the following command:
# # tensorboard --logdir=projector_directory

# # After running the above command, open a web browser and go to the TensorBoard URL to view your embeddings projector.