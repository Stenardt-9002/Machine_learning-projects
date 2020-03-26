#iris dataset

from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf
import pandas as pd
from sklearn import datasets
CSV_COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
SPECIES = ['Setosa','Versicolor','Virginica']


train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe
# iris = datasets.load_iris()\

# print(train)
#
#
# train.head()

train_y = train.pop('Species')
test_y = test.pop('Species')


def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)



# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)



classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,hidden_units=[30, 10], n_classes=3)


classifier.train(input_fn=lambda: input_fn(train, train_y, training=True),steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
print(eval_result)



def input_fn12(features,batch_size = 256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
    pass

features = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
preditc = {}

for feat in features:
    valid1 = True
    while valid1:
        val1 = input(feat + ": ")
        if not val1.isdigit():
            valid1 = False
    preditc[feat] = [float(val1)]

predictions = classifier.predict(input_fn = lambda:input_fn12(preditc))

for pred_dict in predictions:
    class_id1 = pred_dict['class_ids'][0]
    probility = pred_dict['probabilities'][class_id1]

    print("Prediction is {}({:.4f}%) ".format(SPECIES[class_id1],probility))
