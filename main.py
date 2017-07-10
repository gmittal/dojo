# Quora Question Pairs Task
# https://www.kaggle.com/c/quora-question-pairs
# Written by Gautam Mittal
# July 8, 2017

import numpy as np
import tensorflow as tf
import spacy, json

nlp = spacy.load('en')

def load_glove_vecs(path='glove/glove.840B.300d.json'):
    return json.loads(open(path).read())

def pca(mat, n):
    mat = mat - np.mean(mat, axis=0)
    [u,s,v] = np.linalg.svd(mat)
    v = v.transpose()
    v = v[:,:n]
    return np.dot(mat, v)

def get_word_probs(sent_list):
    total = 0.0
    VOCAB = {}
    for sentence in sent_list:
        doc = nlp(unicode(sentence))
        for words in doc:
            total += 1.0
            try:
                VOCAB[words.text.lower()] += 1.0
            except KeyError:
                VOCAB[words.text.lower()] = 1.0
    for k in VOCAB: VOCAB[k] /= total
    return VOCAB

def word_weight(word, a=1.0):
    try word_probs[word]:
        return a / (a + word_probs[word])
    except KeyError:
        return 1

def sent_vec(sent):
    doc = nlp(unicode(sent))
    vec = np.zeros(300)
    for word in doc: vec += word_weight[word.text] * glove[word.text]
    return vec / len(doc)

def feature_vec(sent, fpc):
    vec = sent_vec(sent)
    return vec - direction * direction.T * vec

def vec_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def init():
    questions = ["Hello world", "Who are you, hello"]

    global glove, word_probs, direction
    glove = load_glove_vecs()
    word_probs = get_word_probs(questions)


    direction =

if __name__ == "__main__":
    init()
