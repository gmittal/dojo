# Given a sentence return an embedding vector
import numpy as np
import os, sys, re
from random import randint
import torch
import preprocess

os.chdir('util/InferSent/encoder')

GLOVE_PATH = '../../../glove/glove.840B.300d.txt'
reload(sys)
sys.setdefaultencoding('utf-8')

INFERSENT_TOGGLE = False
glove_model = {}
infersent_model = {}
OOV = []

def words(text): return re.findall(r'\w+', text.lower())

def load_glove():
    f = open(GLOVE_PATH,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    return model

def encode(text):
    global OOV, INFERSENT_TOGGLE
    if INFERSENT_TOGGLE:
        return infersent_model.encode([preprocess.fix(text)])[0]
    else:
        doc = preprocess.fix(text)
        vec = []
        if len(doc) == 0:
            vec.append(np.zeros(300))
        else:
            for word in words(doc):
                try:
                    vec.append(np.asarray(glove_model[word]))
                except KeyError:
                    vec.append(np.zeros(300))
        return np.mean(vec, axis=0)

def init(infersent=False, glove='glove.840B.300d.txt'):
    global glove_model, infersent_model, INFERSENT_TOGGLE, GLOVE_PATH
    GLOVE_PATH = '../../../glove/' + glove
    if infersent:
        INFERSENT_TOGGLE = True
        infersent_model = torch.load('infersent.allnli.pickle')
        infersent_model.set_glove_path(GLOVE_PATH)
        infersent_model.build_vocab_k_words(K=100000)
        infersent_model.use_cuda = True
    else:
        glove_model = load_glove()
