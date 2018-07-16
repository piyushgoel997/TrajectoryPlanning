import numpy as np
import pandas as pd
import cv2
import math


def load_data(data_file):
    """
    loads the training examples into numpy arrays
    :param data_file: a csv file containing the file names of the camera images and the required data
    with the column names CENTER, THROTTLE, ACCELERATION, BRAKE, STEER
    :return: returns an array X containing the images (CENTER) and an array Y containing the rest of the data.
    """
    data = pd.read_csv(data_file)
    size = data.shape[0]  # number of training examples
    X = np.zeros((size, 160, 320, 3))
    Y = np.zeros((size, 4))
    for i in range(size):
        X[i] = cv2.imread("images/" + str(data.loc[i]['CENTER']))
        Y[i, 0] = float(data.loc[i]['THROTTLE'])
        Y[i, 1] = float(data.loc[i]['ACCELERATION'])
        Y[i, 2] = float(data.loc[i]['BRAKE'])
        Y[i, 3] = float(data.loc[i]['STEER'])

    return X, Y


def make_minibatches(X, Y, minibatch_size):
    """
    makes the minibatches of numpy arrays 'X' and 'Y' each of size 'minibatch_size'.
    :param X: a numpy array
    :param Y: a numpy array
    :param minibatch_size: self explanatory (int)
    :return: list of minibatches of X and another list of minibatches of Y.
    """
    X_mb = []
    Y_mb = []

    num_complete_minibatches = math.floor(X.shape[0]/minibatch_size)

    for k in range(int(num_complete_minibatches)):
        X_mb.append(X[k * minibatch_size:(k + 1) * minibatch_size])
        Y_mb.append(Y[k * minibatch_size:(k + 1) * minibatch_size])

    if X.shape[0] % minibatch_size != 0:
        X_mb.append(X[num_complete_minibatches*minibatch_size:])
        Y_mb.append(Y[num_complete_minibatches*minibatch_size:])

    return X_mb, Y_mb
