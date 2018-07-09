import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Embedding
from keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from util.tokenizer_helpers import *

# Set up callbacks
tensorboard = TensorBoard(log_dir='./logs')
early_stopping = EarlyStopping(monitor='val_acc', patience=2)
checkpoint = ModelCheckpoint('save/model.h5',
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max')

train_data = pd.read_csv('data/train.csv')
y = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
train_sent = train_data['comment_text']

tokenizer = Tokenizer(num_words=20000, lower=True)
tokenizer.fit_on_texts(list(train_sent))
train_tokens = tokenizer.texts_to_sequences(train_sent)
save_tokenizer('save/tokenizer.pickle', tokenizer)

train = pad_sequences(train_tokens, maxlen=300)

model = Sequential()
model.add(Embedding(20000, 256, input_length=300))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform"))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
keras.utils.plot_model(model, to_file='save/model.png') # Save a graphical representation of the model

model.fit(train, y, batch_size=32, epochs=20, validation_split=0.1, callbacks=[checkpoint, early_stopping, tensorboard])
