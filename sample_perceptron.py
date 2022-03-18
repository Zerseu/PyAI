import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class BinaryClassifier:
    @staticmethod
    def activation(value: float) -> int:
        return 0 if value <= 0 else 1

    def __init__(self):
        self.w1 = 0.0
        self.w2 = 0.0
        self.bias = 0.0
        self.learning_rate = 0.5

    def train_one(self, x1: float, x2: float, y_true: int):
        output = x1 * self.w1 + x2 * self.w2 + self.bias
        y_pred = BinaryClassifier.activation(output)
        err = y_true - y_pred
        self.bias = self.bias + self.learning_rate * err * 1.0
        self.w1 = self.w1 + self.learning_rate * err * x1
        self.w2 = self.w2 + self.learning_rate * err * x2

    def train(self, data: np.ndarray, epochs: int):
        for epoch in range(epochs):
            for row in data:
                self.train_one(float(row[0]), float(row[1]), int(row[2]))

    def evaluate(self, x1: float, x2: float) -> int:
        output = x1 * self.w1 + x2 * self.w2 + self.bias
        y_pred = BinaryClassifier.activation(output)
        return y_pred


def main():
    x, y_true = datasets.make_blobs(n_samples=100,
                                    n_features=2,
                                    centers=2,
                                    cluster_std=1.05,
                                    random_state=2)
    data = np.hstack((x.astype(dtype=float).reshape((100, 2)),
                      y_true.astype(dtype=float).reshape((100, 1))))

    perceptron = BinaryClassifier()
    perceptron.train(data, 100)

    plt.figure(figsize=(10, 8))
    plt.plot(x[:, 0][y_true == 0], x[:, 1][y_true == 0], 'r^')
    plt.plot(x[:, 0][y_true == 1], x[:, 1][y_true == 1], 'bs')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title('Random Classification Data with 2 Classes')
    x_min = np.min(x[:, 0])
    x_max = np.max(x[:, 0])
    x_plt = np.linspace(x_min, x_max, 100)
    y_plt = -(perceptron.bias + perceptron.w1 * x_plt) / perceptron.w2
    plt.plot(x_plt, y_plt)
    plt.show()


if __name__ == "__main__":
    main()
