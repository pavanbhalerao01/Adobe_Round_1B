# download_model.py
from sentence_transformers import SentenceTransformer
import os

# Specify the model name and the local directory to save it to
model_name = 'all-MiniLM-L6-v2'
save_path = './model'

# Create the directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Download and save the model
print(f"Downloading model '{model_name}' to '{save_path}'...")
model = SentenceTransformer(model_name)
model.save(save_path)
print("Model download complete!")