import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import *
import pickle

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Pre-Processing
x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train.astype('float32')/255.0
y_train = to_categorical(y_train)

model = keras.Sequential(
    [
        Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(filters=16, kernel_size=5, strides=1, activation='relu'),
        MaxPooling2D(pool_size=2, strides=2),
        Flatten(),
        Dense(units=120, activation='relu'),
        Flatten(),
        Dense(units=84, activation='relu'),
        Dense(units=10, activation='sigmoid')
    ]
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
