# Containing all functions for activation functions and their derivatives
import numpy as np
from numpy.typing import NDArray
from typing import Union

def ReLu(z: Union[NDArray[np.float64], float]) -> Union[NDArray[np.float64], float]:
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    ReLU returns the input value if it is positive, otherwise, it returns zero.

    :param z: (numpy.ndarray | float) The input value or array.
    :return: (numpy.ndarray | float) The transformed input with ReLU applied.
    """
    return np.maximum(z, 0)

def ReLu_derivative(z: Union[NDArray[np.float64], float]) -> Union[NDArray[np.bool_], bool]:
    """
    Computes the derivative of the ReLU activation function.

    The derivative is 1 for positive inputs and 0 for negative or zero inputs.

    :param z: (numpy.ndarray | float) The input value or array.
    :return: (numpy.ndarray | bool) The derivative of ReLU (1 for z > 0, otherwise 0).
    """
    return z > 0  # Works nicely because of boolean conversions

def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Applies the softmax activation function.

    Softmax converts input logits into probabilities by exponentiating each value
    and normalizing by the sum of all exponentiated values.

    :param z: (numpy.ndarray) The input array (logits).
    :return: (numpy.ndarray) The probability distribution over input values.
    """
    softmax_vals = np.exp(z) / sum(np.exp(z))
    return softmax_vals