from fastai.vision.all import *
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from PIL import Image


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.constants import DATASET_DIRECTORY, PROJECT_NAME, MODEL_PATH

# URLs of the dataset files
urls = [
    'https://github.com/modestyachts/cifar-10.2/raw/61b0e3ac09809a2351379fb54331668cc9c975c4/cifar102_test.npy',
    'https://github.com/modestyachts/cifar-10.2/raw/61b0e3ac09809a2351379fb54331668cc9c975c4/cifar102_train.npy'
]

# Directory to save the dataset (current working directory)
data_dir = Path(DATASET_DIRECTORY+ "/cifar10_2")

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)
print(data_dir)
# Function to download a file
def download_file(url, save_path):
    print(f'Downloading {url}...')
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)
        print(f'Download complete: {save_path}')
    else:
        print(f'Failed to download {url}. Status code: {response.status_code}')
        response.raise_for_status()

# Download the files if they do not exist
for url in urls:
    file_name = url.split('/')[-1]
    save_path = os.path.join(data_dir, file_name)
    if not os.path.exists(save_path):
        download_file(url, save_path)
    else:
        print(f'File already exists: {save_path}')

# Processing part

train_file_path = os.path.join(data_dir, 'cifar102_train.npy')

# Load the dataset
data = np.load(train_file_path, allow_pickle=True).item()

# Extract arrays
images = data['images']
labels = data['labels']
label_names = data['label_names']

print(f'Number of images: {images.shape[0]}')
print(f'Image shape: {images.shape[1:]}')
print(f'Number of labels: {labels.shape[0]}')
print(f'Label names: {label_names}')

# Base directory to save the images
base_save_dir = os.path.join(data_dir, 'train')

# Create label-based directories if they don't exist
for label_name in label_names:
    label_dir = os.path.join(base_save_dir, label_name)
    os.makedirs(label_dir, exist_ok=True)

# Initialize counters for each label
label_counters = {label_name: 1000 for label_name in label_names}

# Save images in label-based directories
for i in range(images.shape[0]):
    label_name = label_names[labels[i]]
    label_dir = os.path.join(base_save_dir, label_name)
    
    img = Image.fromarray(images[i])
    img.save(os.path.join(label_dir, f'{label_counters[label_name]}_{label_name}.png'))
    
    # Increment the counter for the current label
    label_counters[label_name] += 1

print(f'Images saved to {base_save_dir} in label-based directories.')



# Path to the .npy file
test_file_path = os.path.join(data_dir, 'cifar102_test.npy')

# Load the dataset
data = np.load(test_file_path, allow_pickle=True).item()

# Extract arrays
images = data['images']
labels = data['labels']
label_names = data['label_names']

print(f'Number of images: {images.shape[0]}')
print(f'Image shape: {images.shape[1:]}')
print(f'Number of labels: {labels.shape[0]}')
print(f'Label names: {label_names}')

# Base directory to save the images
base_save_dir = os.path.join(data_dir, 'test')

# Create label-based directories if they don't exist
for label_name in label_names:
    label_dir = os.path.join(base_save_dir, label_name)
    os.makedirs(label_dir, exist_ok=True)

# Initialize counters for each label
label_counters = {label_name: 1000 for label_name in label_names}

# Save images in label-based directories
for i in range(images.shape[0]):
    label_name = label_names[labels[i]]
    label_dir = os.path.join(base_save_dir, label_name)
    
    img = Image.fromarray(images[i])
    img.save(os.path.join(label_dir, f'{label_counters[label_name]}_{label_name}.png'))
    
    # Increment the counter for the current label
    label_counters[label_name] += 1

print(f'Images saved to {base_save_dir} in label-based directories.')

# Create labels.txt
labels_file_path = os.path.join(data_dir, 'labels.txt')
with open(labels_file_path, 'w') as f:
    for label in label_names:
        f.write(f'{label}\n')

print(f'Labels saved to {labels_file_path}')