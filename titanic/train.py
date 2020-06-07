import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/")

TRAIN_SPLIT = 0.8
np.random.seed(0)

# Feature selection and normalization
def cleanData(features):
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
    features['Age1'] = list(map(lambda x: 1 if x <= 16 else 0, features['Age']))
    features['Age2'] = list(map(lambda x: 1 if (x > 16 and x <= 32) else 0, features['Age']))
    features['Age3'] = list(map(lambda x: 1 if (x > 32 and x <= 48) else 0, features['Age']))
    features['Age4'] = list(map(lambda x: 1 if (x > 48 and x <= 64) else 0, features['Age']))
    features['Age5'] = list(map(lambda x: 1 if (x > 64) else 0, features['Age']))
    features['Fare1'] = list(map(lambda x: 1 if x <= 7.91 else 0, features['Fare']))
    features['Fare2'] = list(map(lambda x: 1 if (x > 7.91 and x <= 14.454) else 0, features['Fare']))
    features['Fare3'] = list(map(lambda x: 1 if (x > 14.454 and x <= 31) else 0, features['Fare']))
    features['Fare4'] = list(map(lambda x: 1 if (x > 31) else 0, features['Fare']))
    features['Embarked'] = features['Embarked'].fillna('S')
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
    features['Lady'] = list(map(lambda x: 1 if x == "Lady." else 0, features['Name']))
    features['Countess'] = list(map(lambda x: 1 if x == "Countess." else 0, features['Name']))
    features['Capt'] = list(map(lambda x: 1 if x == "Capt." else 0, features['Name']))
    features['Col'] = list(map(lambda x: 1 if x == "Col." else 0, features['Name']))
    features['Don'] = list(map(lambda x: 1 if x == "Don." else 0, features['Name']))
    features['Sir'] = list(map(lambda x: 1 if x == "Sir." else 0, features['Name']))
    features['Dona'] = list(map(lambda x: 1 if x == "Dona." else 0, features['Name']))
    features = features.drop(columns=['Name', 'Dependents', 'Fare', 'Age', 'Sex', 'Pclass', 'Cabin', 'Ticket', 'Embarked', 'PassengerId', 'Parch', 'SibSp'])
    return features

# Prepare training and validation splits
features = pd.read_csv('data/train.csv')
labels = features['Survived']
features = features.drop(columns=['Survived'])
features = cleanData(features)
msk = np.random.rand(len(features)) < TRAIN_SPLIT

# Training data
train_data = features[msk].values
train_labels = labels[msk].values

# Validation data
val_data = features[~msk].values
val_labels = labels[~msk].values

# Build and train model
model = Sequential()
model.add(Dense(50, input_shape=(35,), activation='relu', kernel_initializer="glorot_uniform"))
model.add(Dense(25, activation='relu', kernel_initializer="glorot_uniform"))
model.add(Dense(1, activation='sigmoid', kernel_initializer="glorot_uniform"))
opt = tf.keras.optimizers.Adam(lr=0.002, beta_1 =0.9, beta_2 = 0.999, decay=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=50, shuffle=True, callbacks=[tensorboard])

test_dat = pd.read_csv("data/test.csv")
result = model.predict(cleanData(test_dat).values)
rta = []
for t in result:
    rta.append(int(round(t[0])))
a = pd.Series(test_dat["PassengerId"], name='PassengerId')
b = pd.Series(rta, name='Survived')

save = pd.DataFrame({'PassengerId':a,'Survived':b})
save.to_csv("submission.csv", index=False)
