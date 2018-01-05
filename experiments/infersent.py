# -*- coding: utf-8 -*-

# Duplicate Question Evaluation Task
# Facebook InferSent Sentence-level Embeddings
# python
# >> import infersent; infersent.init()
# 39% Accuracy for 1500

import numpy as np
import pandas, tqdm, json, os, sys
from random import randint
import torch

os.chdir('InferSent/encoder')

GLOVE_PATH = '../dataset/GloVe/glove.840B.300d.txt'
reload(sys)
sys.setdefaultencoding('utf-8')

train_data = pandas.DataFrame()

def cos_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def evaluate():
    correct = 0.0
    total = 0.0
    for i in tqdm.tqdm(range(len(train_data['question1'][:1500]))):
        if is_ascii(train_data['question1'][i]) and is_ascii(train_data['question1'][i]):
            total += 1.0
            prediction = round(cos_sim(model.encode([train_data['question1'][i]])[0], model.encode([train_data['question2'][i]])[0]))
            if prediction == train_data['is_duplicate'][i]: correct += 1.0
        else:
            continue
    return correct / total

def init():
    global model, train_data
    
    print "Loading Training Data..."
    train_data = pandas.read_csv('../../../data/train.csv')
    questions = list(train_data['question1'].values) + list(train_data['question2'].values)

    print "Building InferSent model..."
    model = torch.load('infersent.allnli.pickle')
    model.set_glove_path(GLOVE_PATH)
    model.build_vocab_k_words(K=100000)
    model.use_cuda = True
   
    print "Accuracy: " + str(evaluate())

if __name__ == "__main__":
    init()
