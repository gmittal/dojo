import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization

tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/")

# Load the training data
train = pd.read_csv('data/train.csv')
train_labels = train['label'].values
train = train.drop(columns=['label'])
train_images = train.values
train_images = train_images.reshape(42000, 28, 28, 1)
train_images = train_images.astype(np.float32)
train_images /= 255
train_labels = tf.keras.utils.to_categorical(train_labels, 10)

# Load the test data
test = pd.read_csv('data/test.csv')
test_images = test.values
test_images = test_images.reshape(28000, 28, 28, 1)
test_images = test_images.astype(np.float32)
test_images /= 255

# Build a huge model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the huge model
model.fit(train_images, train_labels, validation_split=0.2, epochs=10, shuffle=True, batch_size=256, callbacks=[tensorboard])

# Predict some image classes
results = model.predict(test_images)
results = np.argmax(results, axis=1)
results = pd.Series(results,name="Label")
prediction = pd.concat([pd.Series(range(1, 28001), name = "ImageId"),results], axis=1)
prediction.to_csv("submission.csv", index=False)
