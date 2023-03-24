import pickle
from keras.datasets import mnist
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.astype('float32')/255.0
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

test_index = 666
prediction = model.predict(x_test[test_index].reshape(1, 28, 28, 1))
print(f'LeNet Prediction {prediction.argmax()}')
print(f'Correct Answer {y_test[test_index]}')
plt.imshow(x_test[test_index], cmap='gray')
plt.show()
