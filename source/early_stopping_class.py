# Class for early stopping checks and accuracy storing
class EarlyStop:
    """
    Implements early stopping to halt training when validation accuracy stops improving.
    If the accuracy only improves marginally, the training is stopped as well.

    :param training_epochs: (int) Total number of epochs for training.
    :param patience: (int, optional) Number of epochs to wait for improvement before stopping. Default is 10.
    """
    def __init__(self, training_epochs: int, patience=10):
        self.patience = patience
        self.total_epochs = training_epochs
        self.epoch_accuracies = []
        self.penalized_accuracies = []
        self.max_accuracy = 0 # Will hold the best achieved accuracy
        self.best_epoch = 0 # Will hold the epoch connected to the best achieved accuracy
        self.early_stopping = False # Keep track of if early stopping was applied

    @staticmethod
    def apply_accuracy_penalty(current_accuracy: float) -> float:
        """
        Applies a small penalty to the accuracy to prevent minor incremental improvements
        from being considered as meaningful progress.

        :param current_accuracy: (float) The current accuracy value.
        :return: (float) Penalized accuracy, scaled down by a factor of 0.99.
        """
        return 0.995 * current_accuracy

    def check_for_early_stopping(self, current_epoch: int, current_accuracy:float) -> bool:
        """
        Determines whether training should stop based on accuracy stagnation or a high accuracy threshold.

        :param current_epoch: (int) The current epoch index.
        :param current_accuracy: (float) Accuracy of the current epoch.
        :return: (bool) True if training should stop, False otherwise.
        """
        penalized_accuracy = self.apply_accuracy_penalty(current_accuracy)

        self.set_epoch_accuracy(current_accuracy, current_epoch)
        if current_epoch < self.patience:
            return False
        else:
            # Check if new accuracy is worse than previously achieved maximum and enough epochs have passed.
            # Also checks for high accuracy to prevent any overfitting at high accuracies.
            if ((penalized_accuracy <= self.max_accuracy and current_epoch - self.best_epoch >= self.patience) or
                    current_accuracy >= 0.99):
                self.early_stopping = True
                return True
        self.update_best_accuracy(current_accuracy, current_epoch)

    def update_best_accuracy(self, current_accuracy: float, current_epoch: int):
        """
        Updates the best accuracy and corresponding epoch if a new meaningful maximum is found.

        :param current_accuracy: (float) Accuracy of the current epoch.
        :param current_epoch: (int) Current epoch index.
        """
        penalized_accuracy = self.apply_accuracy_penalty(current_accuracy)
        # Compare if the penalized accuracy is still better than the previous unaltered accuracy.
        # Set the unaltered accuracy as the new max to ensure accuracy tracking remains correct.
        # Not setting the penalized acc as the max because this would allow for incremental accuracy improvements.
        if penalized_accuracy > self.max_accuracy:
            self.max_accuracy = current_accuracy
            self.update_best_epoch(current_epoch)

    def update_best_epoch(self, current_epoch: int):
        """
        Updates the best epoch index when a new highest accuracy is achieved.

        :param current_epoch: (int) Epoch at which the best accuracy was recorded.
        """
        self.best_epoch = current_epoch

    def set_epoch_accuracy(self, current_accuracy: float, current_epoch: int):
        """
        Records the accuracy for the current epoch if applicable.

        :param current_accuracy: (float) Accuracy of the current epoch.
        :param current_epoch: (int) Current epoch index.
        :raises IndexError: If the current epoch is out of valid range.
        """
        # Check if value for current_epoch is possible
        if current_epoch > self.total_epochs:
            raise IndexError(f"The value of epoch {current_epoch} is larger than the number of epochs {self.total_epochs}!")
        elif current_epoch < 0:
            raise IndexError(f"The value of epoch {current_epoch} is smaller than zero!")
        else:
            # Append the real and penalized accuracies to both arrays
            self.epoch_accuracies.append(current_accuracy)
            self.penalized_accuracies.append(self.apply_accuracy_penalty(current_accuracy))

    def get_epoch_accuracy(self, current_epoch: int) -> float:
        """
        Retrieves the stored accuracy for a specific epoch.

        :param current_epoch: (int) The epoch index to retrieve accuracy for.
        :return: (float) Accuracy of the specified epoch.
        """
        if 0 <= current_epoch < self.total_epochs:
            return self.epoch_accuracies[current_epoch]
        else:
            raise IndexError(f"The value of epoch {current_epoch} is out of valid range!")
