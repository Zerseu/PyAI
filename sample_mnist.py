import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from function_activation import sigmoid_poly, sigmoid_poly_prime
from function_loss import mse, mse_prime
from layer_activation import ActivationLayer
from layer_fully_connected import FullyConnectedLayer
from neural_network import ArtificialNeuralNetwork

mpl.use('TkAgg')


def main():
    digits = datasets.load_digits()
    n_samples = len(digits.images)

    x_train, x_test, y_train, y_test = train_test_split(digits.images, digits.target, test_size=0.2, shuffle=True)

    x_train = x_train.astype(dtype=float) / 255
    x_train = x_train.reshape((-1, 1, 8 * 8))
    y_train_cat = []
    for y in y_train:
        temp = np.zeros(10)
        temp[y] = 1
        y_train_cat.append(temp)
    y_train_cat = np.array(y_train_cat).reshape((-1, 10))

    x_test = x_test.astype(dtype=float) / 255
    x_test = x_test.reshape((-1, 1, 8 * 8))
    y_test_cat = []
    for y in y_test:
        temp = np.zeros(10)
        temp[y] = 1
        y_test_cat.append(temp)
    y_test_cat = np.array(y_test_cat).reshape((-1, 10))

    net = ArtificialNeuralNetwork()
    net.add(FullyConnectedLayer(8 * 8, 25))
    net.add(ActivationLayer(sigmoid_poly, sigmoid_poly_prime))
    net.add(FullyConnectedLayer(25, 25))
    net.add(ActivationLayer(sigmoid_poly, sigmoid_poly_prime))
    net.add(FullyConnectedLayer(25, 10))
    net.add(ActivationLayer(sigmoid_poly, sigmoid_poly_prime))

    net.use(mse, mse_prime)
    errors = net.fit(x_train, y_train_cat, epochs=100, learning_rate=0.1)

    y_out_cat = net.predict(x_test)
    y_out_cat = np.array(y_out_cat, dtype=float).reshape((-1, 10))
    y_out = np.argmax(y_out_cat, axis=1)
    hits = 0
    for idx in range(len(y_test)):
        if y_out[idx] == y_test[idx]:
            hits += 1
    print('Accuracy is %.2f...' % (hits / len(y_test) * 100))

    plt.plot(errors)
    plt.show()


if __name__ == "__main__":
    main()
