import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

DATA_PATH = "cifar10_data.npz"

def load_data():
    if os.path.exists(DATA_PATH):
        print("Loading data from file...")
        data = np.load(DATA_PATH)
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    else:
        print("Downloading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        np.savez_compressed(DATA_PATH, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    return x_train, y_train, x_test, y_test