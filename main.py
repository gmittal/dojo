# Quora Question Pairs Task
# https://www.kaggle.com/c/quora-question-pairs
# Written by Gautam Mittal
# July 8, 2017

import numpy as np
import tensorflow as tf

def load_glove(path):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model

def vec_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

print vec_sim(np.random.randn(300), np.random.randn(300))
