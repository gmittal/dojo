import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/")

data = pd.read_csv('data/train.csv')
labels = data['Survived']
data['Dependents'] = data['Parch'] + data['SibSp']
features = data.drop(columns=['Name', 'Cabin', 'Age', 'Ticket', 'Embarked', 'PassengerId', 'Survived', 'Parch', 'SibSp'])


# model = Sequential()
# model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
#                  activation='relu',
#                  input_shape=(28, 28, 1)))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(tf.keras.layers.Dropout(0.2))
#
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Dropout(0.2))
#
# model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
#
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
# model.fit(train_images, train_labels, epochs=10, batch_size=256, callbacks=[tensorboard])
# loss, accuracy = model.evaluate(test_images, test_labels)
# print(accuracy)
