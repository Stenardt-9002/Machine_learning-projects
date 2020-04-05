#
# The labels in this dataset are the following:
#
# Airplane
# Automobile
# Bird
# Cat
# Deer
# Dog
# Frog
# Horse
# Ship
# Truck
import time 
a1 = time.time()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt

(tain_imahg,train_label),(test_image,test_labale) = datasets.cifar10.load_data()
print(tain_imahg)

train_images_norm = tain_imahg/255.0
test_images_norm = test_image/255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
IMG_INDEX = 7  # change this to look at other images

plt.imshow(train_images_norm[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_label[IMG_INDEX][0]])
plt.show()


#CNN Architecture
mode1l = models.Sequential()
mode1l.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape = (32,32,3)))
mode1l.add(layers.MaxPooling2D((2,2)))
mode1l.add(layers.Conv2D(64,(3,3),activation = 'relu'))
mode1l.add(layers.MaxPooling2D((2,2)))
mode1l.add(layers.Conv2D(64,(3,3),activation = 'relu'))
print(mode1l.summary())

mode1l.add(layers.Flatten())
mode1l.add(layers.Dense(64,activation = 'relu'))
mode1l.add(layers.Dense(10))



mode1l.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
hist1ory = mode1l.fit(train_images_norm,train_label,epochs = 10,validation_data=(test_images_norm,test_labale))

test_loss,test_acc1 = mode1l.evaluate(test_images_norm,test_labale,verbose=3)
print(test_acc1)
print(time.time()-a1)