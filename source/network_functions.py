# Contains functions for the training of a neural network

# Forward propagation
# Z1 = A0 * W1 + b1
# A1 = f(Z1) | ReLu, Tanh, Sigmoid
# Z2 = A1 * W2 + b2
# A2 = softmax(Z2)

# Backwards propagation
# dZ2 = A2 - Y | Y as one-hot encoding
# dW2 = 1/m dZ2 * A1(T)
# db2 = 1/m * sum(dZ2)
# dW1 = 1/m * dZ1 * X(T)
# dB2 = 1/m * sum(dZ1)

# Updating the network | lr=  learning rate
# W1 = W1 - lr * dW1
# b1 = b1 - lr * db1
# W2 = W2 - lr * dW2
# b2 = b2 - lr * db2


import numpy as np
import source.activation_functions as act


def initial_parameters():
    """
    Initializes the parameters (weights and biases) for a neural network.

    The network consists of two layers with randomly initialized weights and biases.

    :return: Tuple containing weights1, bias1, weights2, and bias2.
    """
    weights1 = np.random.rand(10, 784) - 0.5
    bias1 = np.random.rand(10, 1) - 0.5
    weights2 = np.random.rand(10, 10) - 0.5
    bias2 = np.random.rand(10, 1) - 0.5

    return weights1, weights2, bias1, bias2

def forward_propagation(w1, w2, b1, b2, x):
    """
    Performs forward propagation through the neural network.

    :param w1: (numpy.ndarray) Weights for the first layer.
    :param w2: (numpy.ndarray) Weights for the second layer.
    :param b1: (numpy.ndarray) Bias for the first layer.
    :param b2: (numpy.ndarray) Bias for the second layer.
    :param x: (numpy.ndarray) Input data.
    :return: Tuple containing intermediate activations and outputs (z1, a1, z2, a2).
    """
    z1 = w1.dot(x) + b1
    a1 = act.ReLu(z1)
    z2 = w2.dot(a1) + b2
    a2 = act.softmax(z2)

    return z1, a1, z2, a2

def one_hot_encode_classes(y):
    """
    Converts class labels into a one-hot encoded representation.

    :param y: (numpy.ndarray) Array of class labels.
    :return: numpy.ndarray One-hot encoded class labels.
    """
    # Create empty vector of correct size | assume class labels from 0-9, so add 1 to get 10
    one_hot_encoded_y = np.zeros((y.size, y.max() + 1))
    one_hot_encoded_y[np.arange(y.size), y] = 1 # Set value at correct position
    one_hot_encoded_y = one_hot_encoded_y.transpose()
    return one_hot_encoded_y

def backward_propagation(w2, z1, a1, a2, x, y):
    """
    Performs backpropagation to compute gradients for weights and biases.

    :param w2: (numpy.ndarray) Weights for the second layer.
    :param z1: (numpy.ndarray) Pre-activation values for the first layer.
    :param a1: (numpy.ndarray) Activations from the first layer.
    :param a2: (numpy.ndarray) Output activations from the second layer.
    :param x: (numpy.ndarray) Input data.
    :param y: (numpy.ndarray) True class labels.
    :return: Tuple containing gradients dw1, dw2, db1, db2.
    """
    m = y.size
    one_hot_encoded_y = one_hot_encode_classes(y)
    dz2 = a2 - one_hot_encoded_y
    dw2 = 1/m * dz2.dot(a1.transpose())
    db2 = 1/m * np.sum(dz2)
    dz1 = w2.transpose().dot(dz2) * act.ReLu_derivative(z1)
    dw1 = 1 / m * dz1.dot(x.transpose())
    db1 = 1 / m * np.sum(dz1)
    return dw1, dw2, db1, db2

def update_parameters(w1, w2, dw1, dw2, b1, b2, db1, db2, lr):
    """
    Updates the parameters (weights and biases) using gradient descent.

    :param w1: (numpy.ndarray) Weights for the first layer.
    :param w2: (numpy.ndarray) Weights for the second layer.
    :param dw1: (numpy.ndarray) Gradient of weights for the first layer.
    :param dw2: (numpy.ndarray) Gradient of weights for the second layer.
    :param b1: (numpy.ndarray) Bias for the first layer.
    :param b2: (numpy.ndarray) Bias for the second layer.
    :param db1: (numpy.ndarray) Gradient of biases for the first layer.
    :param db2: (numpy.ndarray) Gradient of biases for the second layer.
    :param lr: (float) Learning rate.
    :return: Updated weights and biases.
    """
    w1 = w1 - lr * dw1
    w2 = w2 - lr * dw2
    b1 = b1 - lr * db1
    b2 = b2 - lr * db2
    return w1, w2, b1, b2,

def get_predictions(a2):
    """
    Returns the predicted class labels based on the highest probability.

    :param a2: (numpy.ndarray) Output activations from the network.
    :return: numpy.ndarray Predicted class labels.
    """
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    """
    Computes the accuracy of the model.

    :param predictions: (numpy.ndarray) Predicted class labels.
    :param y: (numpy.ndarray) True class labels.
    :return: float Accuracy value.
    """
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, iterations, lr):
    """
    Performs gradient descent optimization for the neural network.

    :param x: (numpy.ndarray) Input data.
    :param y: (numpy.ndarray) True class labels.
    :param iterations: (int) Number of training iterations.
    :param lr: (float) Learning rate.
    :return: Trained weights and biases (w1, w2, b1, b2).
    """
    w1, w2, b1, b2 = initial_parameters()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(w1, w2, b1, b2, x)
        dw1, dw2, db1, db2 = backward_propagation(w2, z1, a1, a2, x, y)
        w1, w2, b1, b2 = update_parameters(w1, w2, dw1, dw2, b1, b2, db1, db2, lr)
        if i % 10 == 0:
            print("Iteration:", i)
            print("Accuracy:", get_accuracy(get_predictions(a2), y))
    return w1, w2, b1, b2