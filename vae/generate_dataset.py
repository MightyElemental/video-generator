"""
Used to generate a new dataset consisting of image latent vectors and their captions.
The VAE model must have already been trained.
"""

import os
import json
import pickle
import torch
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
from utils import load_latest_checkpoint

# Import your VAE model definition
from vae_model import VAE

VATEX_JSON = "data/vatex_training.json"
DATA_PATH = "data/train"

OUTPUT_PATH = "data/vae_latent_dataset.pkl"

# Constants
# Must match the VAE's settings
IMG_SIZE = 256
LATENT_DIM = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # Adjust based on your GPU memory
CHANNEL_MULTIPLIER = 3

def save(data: list):
    # Save the latent dataset to a pickle file for later use
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Latent dataset has been saved to '{OUTPUT_PATH}'")

# Transformation (should match the one used during VAE training)
transform = v2.Compose([
    v2.CenterCrop(size=(IMG_SIZE,IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Initialize your VAE model and load the trained weights
model = VAE(img_size=IMG_SIZE, latent_dim=LATENT_DIM, multiplier=CHANNEL_MULTIPLIER).to(DEVICE)
model.eval()

# Load the model's weights
load_latest_checkpoint(model, None)

# Load the pickle file if it exists and save a list of present videoIDs
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, 'rb') as f:
        latent_dataset = pickle.load(f)
    print(f"Loaded {len(latent_dataset)} latent vectors from '{OUTPUT_PATH}'")
else:
    latent_dataset = []

# Load the Vatex training data
with open(VATEX_JSON, 'r', encoding='utf-8') as f:
    vatex_data = json.load(f)

# Filter to only include the json objects that have folders that exist
vatex_data = [
    item
    for item in vatex_data
    if os.path.exists(os.path.join(DATA_PATH, item['videoID']))
]

# Filter to exclude the existing objects in latent_dataset
vatex_data = [
    item
    for item in vatex_data
    if item['videoID'] not in [latent_dataset[i]['videoID'] for i in range(len(latent_dataset))]
]

progress_bar = tqdm(
    vatex_data,
    desc='Processing dataset',
    total=len(vatex_data),
    smoothing=1,
    unit="video",
)
for i, item in enumerate(progress_bar):
    videoID = item['videoID']
    enCap = item['enCap']  # List of English captions

    img_dir = os.path.join(DATA_PATH, videoID)
    if not os.path.exists(img_dir):
        print(f"Image directory {img_dir} does not exist. Skipping.")
        continue  # Skip if directory does not exist

    img_files = sorted(
        [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
    )
    if not img_files:
        print(f"No images found in {img_dir}. Skipping.")
        continue

    latent_vectors = []
    images_batch = []
    for idx, img_file in enumerate(img_files):
        try:
            img = Image.open(img_file).convert('RGB')
            img_tensor = transform(img)
            images_batch.append(img_tensor)
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            continue

        # Process images in batches
        if len(images_batch) == BATCH_SIZE or idx == len(img_files) - 1:
            batch_tensor = torch.stack(images_batch).to(DEVICE)
            with torch.no_grad():
                # Encode the batch of images to get the latent vectors
                mu, logvar = model.encode(batch_tensor)
                latent_vecs = mu.cpu().numpy()  # Shape: (batch_size, latent_dim)
                latent_vectors.extend(latent_vecs)
            images_batch = []

    # Store the data
    data_item = {
        'videoID': videoID,
        'enCap': enCap,
        'latent_vectors': latent_vectors  # List of latent vectors as numpy arrays
    }
    latent_dataset.append(data_item)

    if (i+1) % 100 == 0:
        save(latent_dataset)

# Save the dataset once the processing is complete
save(latent_dataset)
