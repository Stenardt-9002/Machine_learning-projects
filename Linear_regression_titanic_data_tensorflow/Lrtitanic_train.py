#linear regression
# https://www.tensorflow.org/tutorials/estimator/linear
from __future__ import absolute_import,division,print_function,unicode_literals
# absolute means look in path
# print is a function not a statement
# division to give float not integer for python2
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc  #feature column
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#tianic dataset

# dftrain = pd.read_csv('h')
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
# y_train = dftrain.pop
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# print(dftrain.head())
# print(dftrain.describe())
#
# print(dftrain.shape[0])
# print(dfeval.shape[0])
# print(dftrain.loc[0])
# print(dftrain.iloc[0:-1])
# print(y_train[0])

# dftrain.age.hist(bins=89)
# plt.show()
# cols = dftrain.columns.tolist()
# print(cols)
# # print(dftrain.columns[0])
# for i in cols:
#     dftrain.i.hist(bins = 40)
#     plt.show()
#     x = input()


# dftrain.age.hist(bins=20)
# plt.show()
# print(dftrain.sex.value_counts())
# dftrain.sex.value_counts().plot(kind = 'barh')
# plt.show()
#
# dftrain['class'].value_counts().plot(kind = 'barh')
# plt.show()
# print(dftrain['age'].describe())
