import sys, argparse
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from util.tokenizer_helpers import *

# Rebuild saved tokenizer
tokenizer = load_tokenizer('save/tokenizer.pickle')

# Load the test data
test_data = pd.read_csv('data/test.csv')
test_sent = test_data['comment_text']
test_tokens = tokenizer.texts_to_sequences(test_sent)
test = pad_sequences(test_tokens, maxlen=300)

model = load_model('save/model.h5')

results = model.predict(test)
results = np.round(results)
results = results.reshape(results.shape[0])
results = pd.Series(results,name="label")
prediction = pd.concat([pd.Series(range(20800, 26000), name='id'),results], axis=1)
prediction['label'] = prediction['label'].map(int)
prediction.to_csv("submission.csv", index=False)
