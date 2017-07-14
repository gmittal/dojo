# -*- coding: utf-8 -*-

# Quora Question Pairs Task
# https://www.kaggle.com/c/quora-question-pairs
# Written by Gautam Mittal
# July 8, 2017

from collections import Counter
import numpy as np
import tensorflow as tf
import json, pandas, tqdm, re
import preprocess

glove = {}
direction = []
WORDS = {}

def load_glove_vecs(path='glove/glove.840B.300d.json'):
    return json.loads(open(path).read())

# Principal Component Analysis
def pca(mat, n=1):
    mat = mat - np.mean(mat, axis=0)
    [u,s,v] = np.linalg.svd(mat)
    v = v.transpose()
    v = v[:,:n]
    return np.dot(mat, v)

# Simple tokenizer
def words(text): return re.findall(r'\w+', text.lower())

# Probability of words
def P(word):
    return WORDS[word] / sum(WORDS.values())

def word_weight(word, a=1.0):
    try:
        return a / (a + P(word.lower()))
    except KeyError:
        return 1 / (sum(WORDS.values()) + 1)

def sent_vec(sent):
    doc = nlp(unicode(sent))
    vec = [word_weight(word.text) * np.asarray(glove[word.text.lower()]) for word in doc]
    return np.mean(vec, axis=0)

def feature_vec(sent):
    vec = sent_vec(sent)
    return vec - direction * direction.T * vec

def vec_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def init():
    global glove, WORDS, direction

    print "Loading Training & Testing Data..."
    # Load the CSV files
    train_data = pandas.read_csv('data/train.csv')
    test_data = pandas.read_csv('data/test.csv')
    train_data = train_data.fillna('empty')
    test_data = test_data.fillna('empty')

    # Just get the questions
    questions = train_data['question1'].values) + train_data['question2'].values + test_data['question1'].values + test_data['question2'].values

    # Clean everything
    questions = [preprocess.fix(question) for question in questions]

    print "Loading Glove Vectors..."
    glove = load_glove_vecs()

    print "Loading word weights..."
    try:
        WORDS = json.loads(open('data/weights.json').read())
    except IOError:
        WORDS = Counter(words(' '.join(questions)))
        with open('data/weights.json', 'w') as outfile:
                json.dump(WORDS, outfile)

    train_sent_matrix = np.asarray([sent_vec(questions[0])])
    for i in tqdm.tqdm(range(1, len(questions))):
        np.concatenate((train_sent_matrix, np.asarray(sent_vec(questions[i]))))

    print train_sent_matrix.transpose().shape


if __name__ == "__main__":
    init()
