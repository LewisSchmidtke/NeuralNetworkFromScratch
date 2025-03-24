# Initialize relevant imports
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi

from source.network_functions import gradient_descent

# Initialize kaggle api call and download mnist data
api = KaggleApi()
api.authenticate()
api.dataset_download_files('oddrationale/mnist-in-csv', path='data/', unzip=False)

with zipfile.ZipFile('./data/mnist-in-csv.zip', 'r') as zip_ref:
    zip_ref.extractall('./data/')

# Extract training and testing data from kaggle mnist dataset
train_data = pd.read_csv('data/mnist_train.csv')
test_data = pd.read_csv('data/mnist_test.csv')

# Convert to numpy array so we can use in forward prop
train_data = np.array(train_data)
test_data = np.array(test_data)
# Extract shape of data
train_rows, train_columns = train_data.shape
test_rows, test_columns = test_data.shape
# Shuffle data for randomization in training process
np.random.shuffle(train_data)
np.random.shuffle(test_data)
# Transpose data, makes it easier to work with | 1 example = 1 row -> 1 example = 1 column
train_data = train_data.transpose()
test_data = test_data.transpose()

y_train = train_data[0] # Extract class labels
x_train = train_data[1:train_columns] # Pixel data
x_train = x_train / 255 # Normalize values

y_test = test_data[0]
x_test = test_data[1:test_columns]
x_test = x_test / 255


# Set training parameters
EPOCHS = 1000
LEARNING_RATE = 0.1

# Initialize training
w1, w2, b1, b2 = gradient_descent(x_train, y_train, EPOCHS, LEARNING_RATE)
