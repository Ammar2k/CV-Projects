from PIL import Image
import os

# set the path to the directory containing the images
cat_dir = 'PetImages/Cat'
dog_dir = 'PetImages/Dog'

# giving the user permission to remove files
os.chmod('PetImages', 0o755)


def remove_corrupted(file):
    # attempt to open the image with Pillow
    try:
        with Image.open(file) as img:
            # do nothing, image is valid
            pass
    except:
        # if there's an error opening the image, delete the file
        os.remove(file)
        print(f'Removed corrupted image: {file}')


# loop through each image file in the cat directory
for file_name in os.listdir(cat_dir):
    # set the path to the current image file
    file_path = os.path.join(cat_dir, file_name)
    remove_corrupted(file_path)
# loop through each image file in the dog directory
for file_name in os.listdir(dog_dir):
    # set the path to the current image file
    file_path = os.path.join(dog_dir, file_name)
    remove_corrupted(file_path)
