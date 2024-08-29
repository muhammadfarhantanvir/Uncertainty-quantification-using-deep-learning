from fastai.vision.all import *
sys.path.append("./")
from src.constants import DOMAINNET_DIRECTORY
from src.datasets import  get_data_block

# The class is used to generate images to test how augmentations are being applied.
seed = 15
set_seed(seed, reproducible=True)

tgen = torch.Generator().manual_seed(seed)
common_params = {
                'valid_pct': 0.2,
                "use_random_flip": true,
                "use_random_erasing": true,
                "use_randaugment": true,
                "rand_flip_prob": 0.5,
                "random_erasing_prob": 0.5,
                'include_classes': ['lighthouse', 'eye', 'sink', 'dolphin', 'cow', 'banana', 'axe', 'hat', 'fish', 'face'],
                'exclude_classes': None,
                'do_normalize': False,
                'generator':tgen
            }
#Modify the dataset path and directory
# Create the DataBlock
dblock = get_data_block('domainnet', **common_params)

train_path = Path(DOMAINNET_DIRECTORY) / 'real'


# Create the DataLoaders
dls = dblock.dataloaders(train_path, bs=10)

# Get a batch of images
batch = dls.one_batch()

# Get the images and labels from the batch
images, labels = batch

# Define a path to save the augmented images
save_path = Path('augmented_images')
save_path.mkdir(parents=True, exist_ok=True)
#Keeps overwriting the same images
print('Saving images')
for i, img in enumerate(images):
    img = TensorImage(img)
    img = img.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    img = Image.fromarray(img)
    img.save(save_path / f'augmented_image_{i}.png')
print('All images saved')