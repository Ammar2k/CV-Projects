import keras
from keras.datasets import mnist
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the datta
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"y_train shape: {y_train.shape}")
print(f"x_train shape: {x_train.shape}")

# splitting the data into training and validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
)

# defining the model architecture
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    loss=categorical_crossentropy,
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64,
)

_, accuracc = model.evaluate(x_test, y_test, verbose=0)

print(f"Accuracy on the test set is {accuracc}")
