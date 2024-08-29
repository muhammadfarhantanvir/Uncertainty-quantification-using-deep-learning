import tensorflow as tf
import tensorflow_datasets as tfds
import os

def save_images(dataset, save_dir, subset_name, split_name, class_names):
    split_dir = os.path.join(save_dir, subset_name, split_name)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        
    for i, (image, label) in enumerate(dataset):
        label_index = label.numpy()
        label_name = class_names[label_index]
        image = image.numpy()
        
        # Create a directory for each class label name
        label_dir = os.path.join(split_dir, label_name)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # Get the number of existing files to generate the filename
        num_files = len(os.listdir(label_dir))
        filename = os.path.join(label_dir, f'{label_name}_{num_files}.png')
        
        # Save the image using TensorFlow
        tf.keras.preprocessing.image.save_img(filename, image)
        
        if i % 100 == 0:
            print(f'Saved {i} images from {subset_name}/{split_name} split')

def download_domainnet(subset: str, custom_dir: str):
    # Load the dataset with info to get class names
    ds_splits, info = tfds.load(name=f'domainnet/{subset}', split=['train', 'test'], with_info=True, as_supervised=True)
    class_names = info.features['label'].names
    
    # Save images for train and test splits
    for split, dataset in zip(['train', 'test'], ds_splits):
        save_images(dataset, custom_dir, subset, split, class_names)
    
    print(f"Dataset saved at: {custom_dir}")

if __name__ == "__main__":

    #uncomment one by one as simulatenous download will be slow

    download_domainnet(subset="sketch", custom_dir='/home/pgshare/shared/datasets/domainnet')
    # download_domainnet(subset="real", custom_dir='/home/pgshare/shared/datasets/domainnet')
    # download_domainnet(subset="infograph", custom_dir='/home/pgshare/shared/datasets/domainnet')
    # download_domainnet(subset="clipart", custom_dir='/home/pgshare/shared/datasets/domainnet')
    # download_domainnet(subset="quickdraw", custom_dir='/home/pgshare/shared/datasets/domainnet')
    # download_domainnet(subset="painting", custom_dir='/home/pgshare/shared/datasets/domainnet')