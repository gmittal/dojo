# -*- coding: utf-8 -*-

# Quora Question Pairs Task
# https://www.kaggle.com/c/quora-question-pairs
# Written by Gautam Mittal
# July 8, 2017

from collections import Counter
import numpy as np
import tensorflow as tf
import json, pandas, tqdm

glove = {}
word_probs = {}
direction = []

def load_glove_vecs(path='glove/glove.840B.300d.json'):
    return json.loads(open(path).read())

def pca(mat, n):
    mat = mat - np.mean(mat, axis=0)
    [u,s,v] = np.linalg.svd(mat)
    v = v.transpose()
    v = v[:,:n]
    return np.dot(mat, v)

def P(word):
    "Probability of `word`."
    return WORDS[word] / N

def word_weight(word, a=1.0):
    try:
        return a / (a + word_probs[word.lower()])
    except KeyError:
        return 1

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
    global glove, word_probs, direction

    print "Loading Training Data..."
    train_data = pandas.read_csv('data/train.csv')
    questions = list(train_data['question1'].values) + list(train_data['question2'].values)

    print "Loading Glove Vectors..."
    glove = load_glove_vecs()

    print "Loading word weights..."
    try:
        word_probs = json.loads(open('data/weights.json').read())
    except IOError:
        word_probs = get_word_probs(map(lambda x: unicode(x), questions))
        with open('data/weights.json', 'w') as outfile:
                json.dump(word_probs, outfile)

    train_sent_matrix = np.asarray(sent_vec(questions[0]))
    for i in tqdm.tqdm(range(1, len(questions))):
        np.column_stack((train_sent_matrix, np.asarray(sent_vec(questions[i]))))

    print train_sent_matrix.shape


if __name__ == "__main__":
    init()
