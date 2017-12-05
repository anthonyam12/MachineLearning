""" Builds a neural network to predict the 'Happiness Score' of a country in the World Happiness data set.
"""

import pandas as pd
from ann import *

df = pd.read_csv('world_happiness2016.csv')
df = df.reindex(np.random.permutation(df.index))

y = df.loc[:, df.columns == 'Happiness Score'].as_matrix()
x = df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1).as_matrix()

n_classes = 1
ann = Ann(2, [9, 9], n_classes, 7)

# for row in x:
#   print(x)    # prints rows

y_hat = ann.feed_forward(x[0])
ann.back_propagate(y[0], y_hat)

y_hat = ann.feed_forward(x[1])
ann.back_propagate(y[1], y_hat)
