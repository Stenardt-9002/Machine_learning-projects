from tensorflow.keras import datasets, layers, models
import tensorflow as tf
keras = tf.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
import cv2



np.random.seed(3)

X = []
Y = []

with open ('../recordinput.csv', 'r') as f:
    for line in f:
        Y.append(line.rstrip())


all_images = []
img_num = 53
# img_num = 0

with open ('../Recordnoimages.txt', 'r') as f:
    get_max_image = f.read()
# print(get_max_image)
# print((type)(get_max_image))



while img_num < ((int)(get_max_image))-20:
    img = cv2.imread(r'../images/frame_{0}.jpg'.format(img_num), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    # print("Ok1")
    # print(img)
    # print(len(img))
    # print(len(img[0]))
    # print(img[0][0].shape)

    img = img[:, :, np.newaxis]
    # print(img)

    # print(len(img))
    # print(len(img[0]))
    # print(img[0][0].shape)


    all_images.append(img)
    img_num += 1



X = np.array(all_images)

# print(X[0].shape)
# print(X.shape)
# print(len(Y))
# Y = Y[:(int)(get_max_image)-20]
Y = Y[:X[0].shape[0]]
print(len(Y))
print()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=5)

## input image dimensions
img_x, img_y = 300, 500
input_shape = (img_x, img_y, 1)

# convert class vectors to binary class matricies for use in catagorical_crossentropy loss below
# number of action classifications
classifications = 3
y_train = keras.utils.to_categorical(y_train, classifications)
y_test = keras.utils.to_categorical(y_test, classifications)

# CNN model
model = Sequential()
model.add(Conv2D(100, kernel_size=(2, 2), strides=(2, 2), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(classifications, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# tensorboard data callback
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, batch_size=250, epochs=60, validation_data=(x_test, y_test), callbacks=[tbCallBack])

# save weights post training
model.save('fame_ai_weaitss.h5')






