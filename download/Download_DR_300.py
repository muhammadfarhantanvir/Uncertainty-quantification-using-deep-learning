import os
import zipfile

# Set your Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = 'YOUR_USERNAME'
os.environ['KAGGLE_KEY'] = 'YOUR_KEY'

import kaggle
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm

# Authenticate Kaggle API
print("Authenticating Kaggle API...")
kaggle.api.authenticate()

# Create necessary directories
base_path = '/tmp/datasets/dr300_original'
os.makedirs(f'{base_path}/train', exist_ok=True)
os.makedirs(f'{base_path}/test', exist_ok=True)
print(f"Created directories at {base_path}")

# Download all files from the competition
 print("Downloading files from Kaggle...")
 kaggle.api.competition_download_file('diabetic-retinopathy-detection', 'train.zip', path=base_path)
 kaggle.api.competition_download_file('diabetic-retinopathy-detection', 'test.zip', path=base_path)
 kaggle.api.competition_download_file('diabetic-retinopathy-detection', 'trainLabels.csv.zip', path=base_path)

print("Download complete.")

# Unzip files and organize them
 def unzip_file(file_path, dest_dir):
     print(f"Unzipping {file_path} to {dest_dir}...")
     with zipfile.ZipFile(file_path, 'r') as zip_ref:
         zip_ref.extractall(dest_dir)
     print(f"Unzipped {file_path} to {dest_dir}")

# Unzip training and testing files
 for i in range(1, 6):
     unzip_file(f'{base_path}/train.zip.00{i}', f'{base_path}/train')
     unzip_file(f'{base_path}/test.zip.00{i}', f'{base_path}/test')

 unzip_file(f'{base_path}/train.zip', f'{base_path}/train')
 unzip_file(f'{base_path}/test.zip', f'{base_path}/test')
 unzip_file(f'{base_path}/trainLabels.csv.zip', base_path)

print("All files unzipped and organized.")

def preprocess_image(image, size=300):
    """
    Preprocesses an image by resizing, adjusting contrast and brightness, applying a circular crop, and applying Gaussian blur.
    """
    # Step 1: Resize the image
    image = image.resize((size, size), Image.LANCZOS)

    # Step 2: Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast by a factor of 2

    # Step 3: Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.5)  # Increase brightness by a factor of 1.5

    # Step 4: Apply circular crop
    def circular_crop(img):
        np_image = np.array(img)
        if np_image.ndim != 3:
            logger.error("Unexpected image dimensions for circular crop")
            return img
        height, width, _ = np_image.shape
        center_x, center_y = int(width / 2), int(height / 2)
        radius = min(center_x, center_y, int(0.9 * center_x), int(0.9 * center_y))

        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

        circular_mask = dist_from_center <= radius
        np_image[~circular_mask] = 0

        return Image.fromarray(np_image)

    image = circular_crop(image)

    # Step 5: Apply Gaussian blur
    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    return image

def save_images(src_dir, dest_dir, label_dict=None):
    """
    Saves images from the source directory to a specified directory, without organizing by class labels.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    print(f"Listing contents of {src_dir}:")
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            print(os.path.join(root, file))
    
    processed_count = 0
    for root, dirs, files in os.walk(src_dir):
        for image_name in tqdm(files):
            if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
                image_id = image_name.replace('.jpeg', '').replace('.jpg', '').replace('.png', '')
                if label_dict is None or image_id in label_dict:
                    src_image_path = os.path.join(root, image_name)
                    try:
                        image = Image.open(src_image_path)
                        print(f"Opened image {src_image_path}")
                    except Exception as e:
                        print(f"Error opening image {src_image_path}: {e}")
                        continue

                    # Preprocess the image
                    image = preprocess_image(image)

                    # Get the number of existing files to generate the filename
                    num_files = len(os.listdir(dest_dir))
                    filename = os.path.join(dest_dir, f'{num_files}.jpeg')

                    # Save the image with JPEG compression
                    try:
                        image.save(filename, format='JPEG', quality=72)
                        print(f"Saved image {filename}")
                        processed_count += 1
                    except Exception as e:
                        print(f"Error saving image {filename}: {e}")

    print(f"Total processed images from {src_dir}: {processed_count}")

def process_and_save_dr_dataset(base_dir: str, output_dir: str, csv_path: str):
    """
    Processes and saves the Diabetic Retinopathy dataset, applying preprocessing.
    """
    print("Loading CSV file for labels...")
    # Load the CSV file to get the labels
    df = pd.read_csv(os.path.join(base_dir, csv_path))
    label_dict = df.set_index('image')['level'].to_dict()

    print("Saving CSV file in output directory...")
    # Save the CSV file in the output directory
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'trainLabels.csv'), index=False)

    print("Processing and saving training images...")
    # Process training images
    train_dir = os.path.join(base_dir, 'train')
    save_images(train_dir, os.path.join(output_dir, 'train'), label_dict)  # Changed to save directly in 'train'

    print("Processing and saving test images...")
    # Process test images
    test_dir = os.path.join(base_dir, 'test')
    save_images(test_dir, os.path.join(output_dir, 'test'))  # Changed to save directly in 'test'

if __name__ == "__main__":
    base_dir = '/tmp/datasets/dr_original'
    output_dir = '/home/pgshare/shared/datasets/bgraham_dr'
    csv_path = 'trainLabels.csv'
    process_and_save_dr_dataset(base_dir=base_dir, output_dir=output_dir, csv_path=csv_path)
    print("Processing complete.")
