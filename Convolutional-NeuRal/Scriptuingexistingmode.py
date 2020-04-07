#rotate image in notebpopk
# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(
# rotation_range=40,
# width_shift_range=0.2,
# height_shift_range=0.2,
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip=True,
# fill_mode='nearest')

#google model

# https://www.tensorflow.org/tutorials/images/transfer_learning
import time
asrtr = time.time()
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

import tensorflow_datasets as tfds
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
# print(raw_train.shape)
print(type(raw_train))
# print(raw_train)
get_label_name = metadata.features['label'].int2str


# for image,lab in raw_train.take(5):
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(lab))
    
# plt.show()


IMG_SIZE = 160 # All images will be resized to 160x160
# samller

def format_example(image, label):
  """
  returns an image that is reshaped to IMG_SIZE
  """
  image = tf.cast(image, tf.float32) #to float
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label


#calling fuction
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)



# for image,labl in train.take(6):
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(labl))

# plt.show()
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


for img, label in raw_train.take(2):
  print("Original shape:", img.shape)

for img, label in train.take(2):
  print("New shape:", img.shape)


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.summary()

base_model.trainable = False
#stop[ trainnig]
#add your classifier

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = keras.layers.Dense(1)
#0 or 1

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
#create your model
# model.summary()
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# # We can evaluate the model right now to see how it does before training it on our new images
initial_epochs = 5
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)
model.save("dogs_vs_cat1s.h5")  # we can save the model and reload it at anytime in the future
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

print('time taken',time.time()-asrtr)