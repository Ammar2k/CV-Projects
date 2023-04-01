import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import shutil
import random
import glob
from keras.preprocessing.image import ImageDataGenerator

# To Use GPU for training
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define the directories for the training, validation, and test sets
base_dir = 'PetImages'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Create the directories for the training, validation, and test sets if they do not already exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    os.makedirs(valid_dir)
    os.makedirs(test_dir)

    # Loop through each class of images (cats and dogs)
    for dir in ['Cat', 'Dog']:
        source_dir = os.path.join('PetImages', dir)

        # Get a random sample of 650 image files from the source directory
        files = random.sample(glob.glob(os.path.join(source_dir, '*')), 650)

        # Copy each image to the appropriate directory based on its index in the sample
        for i, file in enumerate(files):
            if i < 500:
                dest_dir = os.path.join(train_dir, dir)
            elif i < 600:
                dest_dir = os.path.join(valid_dir, dir)
            else:
                dest_dir = os.path.join(test_dir, dir)

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            file_name = os.path.basename(file)
            new_file_name = '{}_{}'.format(dir, file_name)
            shutil.copyfile(file, os.path.join(dest_dir, new_file_name))


def get_class_label(file_path):
    # extract the class label from the file path
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


train_data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    preprocessing_function=get_class_label
)

batch_size = 32

train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

valid_generator = train_data_gen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

test_generator = train_data_gen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)
