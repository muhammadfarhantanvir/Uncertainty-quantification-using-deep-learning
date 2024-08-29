import os
import zipfile
import kaggle

# Authenticate Kaggle API
print("Authenticating Kaggle API...")
kaggle.api.authenticate()

# Create necessary directories
base_path = '/tmp/datasets/dr_original'
os.makedirs(f'{base_path}/train', exist_ok=True)
os.makedirs(f'{base_path}/test', exist_ok=True)
print(f"Created directories at {base_path}")

# Download all files from the competition
print("Downloading files from Kaggle...")
kaggle.api.competition_download_file('diabetic-retinopathy-detection', 'sample.zip', path=base_path)
kaggle.api.competition_download_file('diabetic-retinopathy-detection', 'sampleSubmission.csv.zip', path=base_path)
kaggle.api.competition_download_file('diabetic-retinopathy-detection', 'trainLabels.csv.zip', path=base_path)

for i in range(1, 6):
    kaggle.api.competition_download_file('diabetic-retinopathy-detection', f'test.zip.00{i}', path=base_path)
    kaggle.api.competition_download_file('diabetic-retinopathy-detection', f'train.zip.00{i}', path=base_path)

print("Download complete.")

# Unzip files and organize them
def unzip_file(file_path, dest_dir):
    print(f"Unzipping {file_path} to {dest_dir}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f"Unzipped {file_path} to {dest_dir}")

unzip_file(f'{base_path}/sample.zip', base_path)
unzip_file(f'{base_path}/sampleSubmission.csv.zip', base_path)
unzip_file(f'{base_path}/trainLabels.csv.zip', base_path)

for i in range(1, 6):
    unzip_file(f'{base_path}/test.zip.00{i}', f'{base_path}/test')
    unzip_file(f'{base_path}/train.zip.00{i}', f'{base_path}/train')

print("All files unzipped and organized.")

print("Starting Processing...")
from PIL import Image as PILImage
import os

def preprocess_and_save_image(img_path, dest_img_path):
    # Load image
    img = PILImage.open(img_path)
    
    # Resize image
    img = img.resize((250, 250))
    
    # Crop and pad image (simple cropping in this case)
    img = img.crop((0, 0, 250, 250))
    
    # Save image with 72 DPI
    img.save(dest_img_path, dpi=(72, 72))

def process_directory(src_path, dest_path):
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(img_path, src_path)
                dest_img_path = os.path.join(dest_path, relative_path)
                
                # Ensure destination directory exists
                os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
                
                # Process and save the image
                preprocess_and_save_image(img_path, dest_img_path)

# Define source and destination directories
src_test_path = '/tmp/datasets/dr_original/test'
src_train_path = '/tmp/datasets/dr_original/train'

dest_test_path = '/tmp/datasets/dr250/test'
dest_train_path = '/tmp/datasets/dr250/train'

# Process directories
process_directory(src_test_path, dest_test_path)
process_directory(src_train_path, dest_train_path)

print("All files processed.")

