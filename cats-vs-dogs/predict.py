import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score


base_dir = 'PetImages'
test_dir = os.path.join(base_dir, 'test')


test_images = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_dir, target_size=(224, 224), classes=['Cat', 'Dog'],
                         batch_size=10, shuffle=False)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(test_images)
accuracy = accuracy_score(test_images.classes, np.argmax(prediction, axis=-1))
print(f'Accuracy on the test set is {accuracy}.')
