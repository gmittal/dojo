import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/")

TRAIN_SPLIT = 0.7
np.random.seed(0)

data = pd.read_csv('data/train.csv')
labels = data['Survived']
data['Dependents'] = data['Parch'] + data['SibSp']
features = data.drop(columns=['Name', 'Cabin', 'Age', 'Ticket', 'Embarked', 'PassengerId', 'Survived', 'Parch', 'SibSp'])

# Data normalization
features['Sex'] = list(map(lambda x: 1 if x == 'female' else 0, features['Sex'])) # Sex should be mapped to binary value

mean = features.mean(axis=0)
features -= mean
std = features.std(axis=0)
features /= std

# Prepare training and validation splits
msk = np.random.rand(len(features)) < TRAIN_SPLIT

# Training data
train_data = features[msk].values
train_labels = labels[msk].values

# Validation data
val_data = features[~msk].values
val_labels = labels[~msk].values

print(train_data)
print(train_labels)

model = Sequential()
model.add(Dense(128, input_shape=(4,), activation='sigmoid'))
model.add(Dense(1, activation=tf.nn.softmax))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, batch_size=256, callbacks=[tensorboard])
# loss, accuracy = model.evaluate(test_images, test_labels)
# print(accuracy)
