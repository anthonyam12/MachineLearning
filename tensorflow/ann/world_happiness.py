""" Builds a neural network to predict the 'Happiness Score' of a country in the World Happiness data set.
"""

# Pandas is a popular data analysis library for Python: https://pandas.pydata.org/
# Numpy is a scientific computing package for Python: http://www.numpy.org/
import pandas as pd
import numpy as np

# import the ANN class we've created
from ann import *

# reads data from the world happiness CSV. Data from Kaggle: https://www.kaggle.com/unsdsn/world-happiness
df = pd.read_csv('world_happiness2016.csv')

# since the data was ordered by happiness, we use this call to randomize the row order
df = df.reindex(np.random.permutation(df.index))

# Happiness Score is a continuous variables making this a regression problem
y = df.loc[:, df.columns == 'Happiness Score']

# drop some columns that we won't use in the network. These were all categorical
x = df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1)

# split dataset into test/train, just split in half since the rows are randomized above
train_x = x.iloc[:75, :].as_matrix()
train_y = y.iloc[:75, :].as_matrix()

x = x.iloc[75:, :].as_matrix()
y = y.iloc[75:, :].as_matrix()

n_classes = 1
# Create the ANN with 10 hidden layers of 20 nodes each, n_classes = 1 output, and x.shape[1] is the number of columns
# in the 'x portion' of the dataset.
ann = ANN(10, [20 for _ in range(10)], n_classes, x.shape[1])
ann.train(train_x, train_y)


