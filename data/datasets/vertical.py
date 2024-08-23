import numpy as np
import matplotlib.pyplot as plt

def create_data(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        X[ix] = np.c_[np.random.randn(samples) * .1 + (class_number) / 3, np.random.randn(samples) * .1 + 0.5]
        y[ix] = class_number
    return X, y

def print_data():
    X, y = create_data(samples=100, classes=3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
    plt.show()
