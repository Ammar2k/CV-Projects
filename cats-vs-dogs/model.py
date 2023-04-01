import tensorflow as tf
import os
import shutil
import random
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.regularizers import l2
import pickle

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

train_images = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_dir, target_size=(224, 224), classes=['Cat', 'Dog'], batch_size=10)
valid_images = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_dir, target_size=(224, 224), classes=['Cat', 'Dog'], batch_size=10)
test_images = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_dir, target_size=(224, 224), classes=['Cat', 'Dog'], batch_size=10, shuffle=False)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])

model.fit(x=train_images, validation_data=valid_images, epochs=10, verbose=2)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
