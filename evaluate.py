import sys, argparse
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from util.tokenizer_helpers import *

# Parse command line arguments
parser = argparse.ArgumentParser(description='Try the toxic comment classifier.')
parser.add_argument('-t', action='store', dest='test', default=True, help='Make predictions on test.csv and output a submission.csv')
parser.add_argument('-c', action='store', dest='comment', type=str, help='Classify an example comment')
args = parser.parse_args()

# Rebuild saved tokenizer
tokenizer = load_tokenizer('save/tokenizer.pickle')

# Load the test data
test_data = pd.read_csv('data/test.csv')
test_sent = test_data['comment_text']
test_tokens = tokenizer.texts_to_sequences(test_sent)
test = pad_sequences(test_tokens, maxlen=300)

model = load_model('save/model.h5')

if args.comment:
    args.test = False
    comment_tokens = tokenizer.texts_to_sequences([args.comment])
    comment_vec = pad_sequences(comment_tokens, maxlen=300)
    predictions = model.predict(comment_vec)[0]
    print('Toxic: {0:.2f}%'.format(predictions[0] * 100))
    print('Severe Toxic: {0:.2f}%'.format(predictions[1] * 100))
    print('Obscene: {0:.2f}%'.format(predictions[2] * 100))
    print('Threat: {0:.2f}%'.format(predictions[3] * 100))
    print('Insult: {0:.2f}%'.format(predictions[4] * 100))
    print('Identity Hate: {0:.2f}%'.format(predictions[5] * 100))

if not args.test: # Terminate the program early if we're not creating test predictions
    sys.exit()

submission = pd.read_csv('data/sample_submission.csv')
y_pred = model.predict(test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)
