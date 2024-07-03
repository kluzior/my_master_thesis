import os
from prepare_images import process_images_in_folder
from teaching import train_model, save_model
from clasification import classify_new_images

# Define paths
reference_folder = 'neural_network_test/new/refs2/'
results_folder = 'neural_network_test/new/cropped2/'
model_file_path = 'mlp_model.joblib'
new_images_folder = 'neural_network_test/new/images/'

# Step 1: Prepare images
process_images_in_folder(reference_folder, results_folder)

# Step 2: Train model
train_model(results_folder, model_file_path)

# Step 3: Classify new images
classify_new_images(new_images_folder, model_file_path)
