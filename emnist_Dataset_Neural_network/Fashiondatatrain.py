# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# fshio_ddat = keras.datasets.fashion_mnist
# # print(type(fshio_ddat))
#
# (train_img_data ,train_img_lbl),(test_img_data,test_lbl_data) = fshio_ddat.load_data()
#
# # print(type(train_img_data))
# print(train_img_data.shape)
# print(train_img_data[0:])


# https://www.tensorflow.org/tutorials/keras/classification
#/...SCRIPT usages Fashion MNIST...../#



from __future__ import absolute_import, division, print_function, unicode_literals
import time
# TensorFlow and` tf.keras
strt1time = time.time()
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
# print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

# print(help(fashion_mnist))
# print(fashion_mnist.load_data())
(train_image , trina_labels ),(test_images,test_lables) = fashion_mnist.load_data()
print(type(train_image))
print(train_image.shape)
# print(train_image)
# print(test_images)
# print(trina_labels)
# print(test_lables[0])
# print(len(train_image))
# x1 = np.asarray(train_image[2])
# print(type(x1))
# print(x1.shape)
# plt.imshow(x1,interpolation="nearest")
# plt.show()
#
# for i in range(1000):
#     x1 = np.asarray(train_image[i])
#     plt.imshow(x1,interpolation="nearest")
#     plt.show()
#     xerf = input()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# for i1j in train_image:
#     print(i1j)
#     x = input()
#     os.system("clear")
# processing figurew

# plt.figure()
# plt.imshow(train_image[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


#rescaling images

new_train_images = train_image/255.0

new_test_imahes = test_images/250.0


plt.figure(figsize=(10,10))
for iq in range(25):
    plt.subplot(5,5,iq+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # plt.imshow(train_image[iq],cmap=plt.cm.binary)
    # plt.imshow(train_image[iq])
    plt.imshow(new_test_imahes[iq])


    plt.xlabel(class_names[trina_labels[iq]])

plt.show()
mode1l = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),keras.layers.Dense(128, activation='relu'), keras.layers.Dense(10)])
#
# The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels). Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
#
# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons). The second (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.

mode1l.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#
# def compile(self,
#             optimizer='rmsprop',
#             loss=None,
#             metrics=None,
#             loss_weights=None,
#             sample_weight_mode=None,
#             weighted_metrics=None,
#             target_tensors=None,
#             distribute=None,
#             **kwargs):



mode1l.fit(new_train_images,trina_labels,epochs=10)


# def fit(self,
#         x=None,
#         y=None,
#         batch_size=None,
#         epochs=1,
#         verbose=1,
#         callbacks=None,
#         validation_split=0.,
#         validation_data=None,
#         shuffle=True,
#         class_weight=None,
#         sample_weight=None,
#         initial_epoch=0,
#         steps_per_epoch=None,
#         validation_steps=None,
#         validation_freq=1,
#         max_queue_size=10,
#         workers=1,
#         use_multiprocessing=False,
#         **kwargs):
# new_train_images = train_image/255.0
#
# new_test_imahes = test_images/250.0


# testloss,test_acc = mode1l.evaluate(test_images,test_lables,verbose=2) #verboae will shwo you details
# testloss,test_acc = mode1l.evaluate(new_test_imahes,test_lables,verbose=2) #verboae will shwo you details
testloss,test_acc = mode1l.evaluate(new_test_imahes,test_lables) #verboae will shwo you details

print("\n Accuracy ",test_acc)


proba_model = tf.keras.Sequential([mode1l,tf.keras.layers.Softmax()])
# predictionsontest
predictiononprob = proba_model.predict(new_test_imahes)
print(len(predictiononprob))


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictiononprob[i], test_lables,new_test_imahes)# test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictiononprob[i],  test_lables)
plt.show()


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictiononprob[i], test_lables, new_test_imahes)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictiononprob[i], test_lables)
plt.tight_layout()
plt.show()
print("\nTIme taiken ",time.time()-strt1time)
