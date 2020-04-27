# https://www.tensorflow.org/tutorials/images/transfer_learning
import os

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds 
tfds.disable_progress_bar()
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)


# print(raw_train)
# print(type(raw_train))
# print(raw_validation)
# print(type(raw_validation))
# print(raw_test)
# print(type(raw_test))

get_label_name = metadata.features['label'].int2str

# for image, label in raw_train.take(2):
#   plt.figure()
#   plt.imshow(image)
#   plt.title(get_label_name(label))


# plt.show()


IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label


train1 = raw_train.map(format_example)
# train = raw_train.map(format_example)
validation1 = raw_validation.map(format_example)
test1 = raw_test.map(format_example)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000


trained_bitches = train1.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_bitches = validation1.batch(BATCH_SIZE)
test_bitches = test1.batch(BATCH_SIZE)

for image_batch, label_batch in trained_bitches.take(1):
   pass

# print(image_batch.shape)


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

feature_batch = base_model(image_batch)
# print(feature_batch.shape)

base_model.trainable = False #freeze

# print(base_model.summary()) 

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)



prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer

])



base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# print(model.summary())
# print(len(model.trainable_variables))


initial_epochs = 10
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_bitches, steps = validation_steps)


print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
#completely random results in almost 50 %

# 20/20 [==============================] - 6s 275ms/step - loss: 0.8441 - accuracy: 0.4688
# initial loss: 0.84

# initial accuracy: 0.47


filmodel = model.fit(trained_bitches,
                    epochs=initial_epochs,
                    validation_data=validation_bitches)


acc = filmodel.history['accuracy']
val_acc = filmodel.history['val_accuracy']

loss = filmodel.history['loss']
val_loss = filmodel.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
