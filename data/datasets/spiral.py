import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def create_data(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype=np.uint8)
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y

def print_data():
    X, y = create_data(100, 3)

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
    plt.show()
