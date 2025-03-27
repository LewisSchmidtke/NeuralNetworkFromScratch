# Functions for visualizing data.
import numpy as np
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt


def plot_training_accuracy(training_accuracy: List[float] | np.ndarray, save_image: bool = False,
                           save_location: str = "") -> None:
    """
    Plots the training accuracy over epochs.

    This function creates a line plot of the training accuracy over epochs. If the `save_image` parameter is `True`,
    the plot is saved to the specified location. Otherwise, the plot is displayed.

    :param training_accuracy: (List or Array) A list or array of accuracy values recorded during training.
    :param save_image: (bool, default: False) Whether to save the plot as an image.
    :param save_location: (str, default: "") The file path where the plot image will be saved, if `save_image` is `True`.

    :raises ValueError: If `save_image` is `True` and `save_location` is an empty string.

    :return: None
    """
    # Check for empty string if plot is supposed to be saved. Removes plot computing if not valid
    if save_image and not save_location:
        raise ValueError("When saving the plot a valid save location is needed.")

    # Plot accuracy as a single line
    sns.lineplot(training_accuracy, color="orange", label="Training Accuracy").set(
        title="Training Accuracy over Epochs", xlabel="Epochs", ylabel="Accuracy"
    )

    # Return early if plot is not to be saved
    if not save_image:
        plt.show()
        return

    # Saves image of plot at desired location
    plt.savefig(save_location)
    plt.close()
