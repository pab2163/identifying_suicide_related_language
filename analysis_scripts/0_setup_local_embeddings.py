import pickle
import sys
from sentence_transformers import SentenceTransformer


'''
# Setup for low lexicon similarity analyses

This script prepares the environment for running low lexicon similarity analyses.
It performs two main tasks:

1. **Creates a blank pickle file**:
   - This serves as a placeholder for storing text embeddings later in the pipeline.
   - It ensures that downstream scripts have a valid pickle file to load from.

2. **Downloads a SentenceTransformer model locally**:
   - The model is saved to the `models/` directory to enable fully local embedding computation
     (avoiding cloud-based calls during runtime).

Usage notes:
-------------
- Ensure that the directory `data/embeddings/` exists before running this script.
- The pickle file will be empty initially, to be populated later with text embeddings.
- The model used here (`all-MiniLM-L6-v2`) is a lightweight, general-purpose embedding model.
'''


# ----------------------------------------------------
# Step 1: Create a blank pickle file for embeddings
# ----------------------------------------------------

# Define an empty object (dictionary by default) that will later store embeddings.
empty_object = {}  

# Specify the output path for the blank pickle file.
file_path = 'data/embeddings/stored_embeddings.pickle'

# Create (or overwrite) the pickle file in binary write mode.
with open(file_path, 'wb') as file:
    pickle.dump(empty_object, file)

print(f"Blank pickle file created: {file_path}")

# Verify that the pickle file can be successfully read back in.
with open('data/embeddings/stored_embeddings.pickle', 'rb') as file:
    ctest = pickle.load(file)



# ----------------------------------------------------
# Step 2: Download and store SentenceTransformer model
# ----------------------------------------------------

'''
Download the model locally to ensure embedding generation runs fully offline.

Model details:
- Name: 'all-MiniLM-L6-v2'
- Source: SentenceTransformers (HuggingFace)
- Typical use: General-purpose text embeddings (fast and lightweight)
'''

# Define local save path for the model.
modelPath = 'models/all-MiniLM-L6-v2-local'

# Load the pretrained model (downloads automatically if not cached).
model = SentenceTransformer('all-MiniLM-L6-v2')

# Save the model locally to the specified directory.
model.save(modelPath)

print(f"Model downloaded and saved locally at: {modelPath}")
