import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/")

TRAIN_SPLIT = 0.9
np.random.seed(0)

features = pd.read_csv('data/train.csv')
labels = features['Survived']
features['Dependents'] = features['Parch'] + features['SibSp']
features['Siblings'] = list(map(lambda x: 1 if x > 0 else 0, features['SibSp']))
features['Parents'] = list(map(lambda x: 1 if x > 0 else 0, features['Parch']))
features['Female'] = list(map(lambda x: 1 if x == 'female' else 0, features['Sex'])) # Sex should be mapped to binary value
features['Male'] = list(map(lambda x: 1 if x == 'male' else 0, features['Sex']))
features['Alone'] = list(map(lambda x: 0 if x == 0 else 1, features['Dependents']))
features['Class1'] = list(map(lambda x: 1 if x == 1 else 0, features['Pclass']))
features['Class2'] = list(map(lambda x: 1 if x == 2 else 0, features['Pclass']))
features['Class3'] = list(map(lambda x: 1 if x == 3 else 0, features['Pclass']))
avg_age = np.mean(features['Age'][np.isfinite(features['Age'])].values)
features['Age'] = list(map(lambda x: avg_age if np.isnan(x) else x, features['Age']))
features['Age1'] = list(map(lambda x: 1 if x <= 14 else 0, features['Age']))
features['Age2'] = list(map(lambda x: 1 if (x > 14 and x <= 28) else 0, features['Age']))
features['Age3'] = list(map(lambda x: 1 if (x > 28 and x <= 38) else 0, features['Age']))
features['Age4'] = list(map(lambda x: 1 if (x > 38) else 0, features['Age']))
features['Fare1'] = list(map(lambda x: 1 if x <= 7.5 else 0, features['Fare']))
features['Fare2'] = list(map(lambda x: 1 if (x > 7.5 and x <= 14.45) else 0, features['Fare']))
features['Fare3'] = list(map(lambda x: 1 if (x > 14.45 and x <= 31) else 0, features['Fare']))
features['Fare4'] = list(map(lambda x: 1 if (x > 31) else 0, features['Fare']))
features['Embarked1'] = list(map(lambda x: 1 if x == "S" else 0, features['Embarked']))
features['Embarked2'] = list(map(lambda x: 1 if x == "C" else 0, features['Embarked']))
features['Embarked3'] = list(map(lambda x: 1 if x == "Q" else 0, features['Embarked']))
features['Name'] = list(map(lambda x: x.split(",")[1].split()[0], features['Name']))
features['Mr'] = list(map(lambda x: 1 if x == "Mr." else 0, features['Name']))
features['Miss'] = list(map(lambda x: 1 if x == "Miss." else 0, features['Name']))
features['Mrs'] = list(map(lambda x: 1 if x == "Mrs." else 0, features['Name']))
features['Master'] = list(map(lambda x: 1 if x == "Master." else 0, features['Name']))
features['Dr'] = list(map(lambda x: 1 if x == "Dr." else 0, features['Name']))
features['Major'] = list(map(lambda x: 1 if x == "Major." else 0, features['Name']))
features['Mme'] = list(map(lambda x: 1 if x == "Mme." else 0, features['Name']))
features['Rev'] = list(map(lambda x: 1 if x == "Rev." else 0, features['Name']))
features = features.drop(columns=['Name', 'Dependents', 'Fare', 'Age', 'Sex', 'Pclass', 'Cabin', 'Ticket', 'Embarked', 'PassengerId', 'Survived', 'Parch', 'SibSp'])

# Prepare training and validation splits
msk = np.random.rand(len(features)) < TRAIN_SPLIT

# Training data
train_data = features[msk].values
train_labels = labels[msk].values

# Validation data
val_data = features[~msk].values
val_labels = labels[~msk].values

model = Sequential()
model.add(Dense(50, input_shape=(27,), activation='relu', kernel_initializer="glorot_uniform"))
model.add(Dense(200, activation='relu', kernel_initializer="glorot_uniform"))
model.add(Dense(200, activation='relu', kernel_initializer="glorot_uniform"))
model.add(Dense(50, activation='relu', kernel_initializer="glorot_uniform"))
model.add(Dense(20, activation='relu', kernel_initializer="glorot_uniform"))
model.add(Dense(1, activation='sigmoid', kernel_initializer="glorot_uniform"))
opt = tf.keras.optimizers.Adam(lr=0.002, beta_1 =0.9, beta_2 = 0.999, decay=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=20, shuffle=True, callbacks=[tensorboard])
# loss, accuracy = model.evaluate(test_images, test_labels)
# print(accuracy)
