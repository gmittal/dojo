import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/")

TRAIN_SPLIT = 0.75
np.random.seed(0)

features = pd.read_csv('data/train.csv')
labels = features['Survived']
features['Dependents'] = features['Parch'] + features['SibSp']
features['Female'] = list(map(lambda x: 1 if x == 'female' else 0, features['Sex'])) # Sex should be mapped to binary value
features['Male'] = list(map(lambda x: 1 if x == 'male' else 0, features['Sex']))
features['Alone'] = list(map(lambda x: 1 if x == 0 else 0, features['Dependents']))
features['Class1'] = list(map(lambda x: 1 if x == 1 else 0, features['Pclass']))
features['Class2'] = list(map(lambda x: 1 if x == 2 else 0, features['Pclass']))
features['Class3'] = list(map(lambda x: 1 if x == 3 else 0, features['Pclass']))
median_age = np.median(features['Age'][np.isfinite(features['Age'])].values)
features['Age'] = list(map(lambda x: median_age if np.isnan(x) else x, features['Age']))
features = features.drop(columns=['Name', 'Sex', 'Pclass', 'Cabin', 'Ticket', 'Embarked', 'PassengerId', 'Survived', 'Parch', 'SibSp'])


print(features)

# Prepare training and validation splits
msk = np.random.rand(len(features)) < TRAIN_SPLIT

# Training data
train_data = features[msk].values
train_labels = labels[msk].values

# Validation data
val_data = features[~msk].values
val_labels = labels[~msk].values

model = Sequential()
model.add(Dense(100, input_shape=(6,), activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, shuffle=True, callbacks=[tensorboard])
# loss, accuracy = model.evaluate(test_images, test_labels)
# print(accuracy)
