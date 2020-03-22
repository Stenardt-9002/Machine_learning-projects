# #linear regression
# # https://www.tensorflow.org/tutorials/estimator/linear
# from __future__ import absolute_import,division,print_function,unicode_literals
# # absolute means look in path
# # print is a function not a statement
# # division to give float not integer for python2
# from IPython.display import clear_output
# from six.moves import urllib
# import tensorflow.compat.v2.feature_column as fc  #feature column
# import tensorflow as tf

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# #tianic dataset

# # dftrain = pd.read_csv('h')
# dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
# dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
# # y_train = dftrain.pop
# y_train = dftrain.pop('survived')
# y_eval = dfeval.pop('survived')

# # print(dftrain.head())
# # print(dftrain.describe())
# #
# # print(dftrain.shape[0])
# # print(dfeval.shape[0])
# # print(dftrain.loc[0])
# # print(dftrain.iloc[0:-1])
# # print(y_train[0])

# # dftrain.age.hist(bins=89)
# # plt.show()
# # cols = dftrain.columns.tolist()
# # print(cols)
# # # print(dftrain.columns[0])
# # for i in cols:
# #     dftrain.i.hist(bins = 40)
# #     plt.show()
# #     x = input()


# # dftrain.age.hist(bins=20)
# # plt.show()
# # print(dftrain.sex.value_counts())
# # dftrain.sex.value_counts().plot(kind = 'barh')
# # plt.show()
# #
# # dftrain['class'].value_counts().plot(kind = 'barh')
# # plt.show()
# # print(dftrain['age'].describe())
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')


# # cols = dftrain.columns.tolist()
# # print(cols)
# # CATE_COLS = []
# # NUM_COLS = []
# # for i in cols:
# #     print(i)
# #     print(dftrain[i].dtypes)


# CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
#                        'embark_town', 'alone']
# NUMERIC_COLUMNS = ['age', 'fare']

# feature_columns = []
# for feature_name in CATEGORICAL_COLUMNS:
#   vocabulary = dftrain[feature_name].unique()
#   feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
#   # print()
#   # print(vocabulary)
#   # print(feature_columns)
#   # x = input()

# # print(feature_columns)
# # print()

# for feature_name in NUMERIC_COLUMNS:
#   feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
# print(feature_columns)  #feature columns
# print()

# # for i in feature_columns:
# #     print(i)


#GPU development
# https://www.tensorflow.org/tutorials/estimator/linear
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf


# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# print(dftrain)
# print(dfeval)
# print(y_train)

# print(dftrain.head())
# print(dftrain.describe())
# print(dftrain.shape[0])
# print(dfeval.shape[0])
# print(dftrain.loc[0])
# print(y_train.loc[0])

# dftrain.age.hist(bins=80)
# dftrain.sex.value_counts().plot(kind='barh')
# dftrain['class'].value_counts().plot(kind='barh')

# plt.show()
#understood datset

#create proper category

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
  # print(vocabulary)
  # z = input()



for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# print(vocabulary)
# print(feature_columns[1])



def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    #conversion given tensor are sliced
    if shuffle:
      ds = ds.shuffle(1000) #shffing
    ds = ds.batch(batch_size).repeat(num_epochs) #repeat in different blocks
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
# ds = make_input_fn(dftrain, y_train, batch_size=10)()
# for feature_batch, label_batch in ds.take(1):
#   print('Some feature keys:', list(feature_batch.keys()))
#   print()
#   print('A batch of class:', feature_batch['class'].numpy())
#   print()
#   print('A batch of Labels:', label_batch.numpy())
# age_column = feature_columns[7]
# tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()


linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
# print(linear_est)
# print(result)
# tf.data.Dataset




#predict
result1 = list(linear_est.predict(eval_input_fn))
print(result1[0]["probabilities"]) #0 not survived
print(dfeval.loc[0])
print(y_eval.loc[0])
print("OOOXVDOOXDVODSOV")
result1 = list(linear_est.predict(eval_input_fn))
print(result1[6]["probabilities"]) #6 not survived
print(dfeval.loc[6])
print(y_eval.loc[6])

