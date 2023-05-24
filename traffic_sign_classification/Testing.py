import os
import random

import cv2
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# Load the architecture
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the weights
loaded_model.load_weights("model_weights.h5")

# Print the model summary
print(loaded_model.summary())


loaded_model.compile(
    Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

path = "myData"  # folder with all the class folders
labelFile = "labels.csv"  # file with all names of classes
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))


# Importing of the Images

count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)
y_test = classNo
y_test = to_categorical(y_test, noOfClasses)


# PREPROCESSING THE IMAGES


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)  # CONVERT TO GRAYSCALE
    img = equalize(img)  # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img / 255  # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img


# TO IRETATE AND PREPROCESS ALL IMAGES
X_test = np.array(list(map(preprocessing, images)))
cv2.imshow(
    "GrayScale Images", X_test[random.randint(0, len(X_test) - 1)]
)  # TO CHECK IF THE TRAINING IS DONE PROPERLY


# ADD A DEPTH OF 1
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


_, accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", accuracy)
