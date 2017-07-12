# -*- coding: utf-8 -*-

# Quora Duplicate Question Evaluation Task
# Unweighted Bag of Words Model using One-Hot Word Vector Space Representations

import numpy as np
import spacy, pandas, tqdm, sys

reload(sys)
sys.setdefaultencoding('utf-8')
nlp = spacy.load('en')

VOCAB = []
train_data = pandas.DataFrame()

def one_hot(n):
    v = np.zeros(len(VOCAB))
    v[n] = 1
    return v

def build_vocab(text):
    global VOCAB
    VOCAB += map(lambda x: str(x).lower(), list(nlp(unicode(text))))
    VOCAB = list(set(VOCAB))

def doc_vec(text):
    doc = nlp(unicode(text))
    vec = np.zeros(len(VOCAB))
    for word in doc: vec += one_hot(VOCAB.index(str(word.text).lower()))
    return vec

def cos_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def evaluate():
    correct = 0.0
    for i in tqdm.tqdm(range(len(train_data['question1'][:1000]))):
        prediction = round(cos_sim(doc_vec(train_data['question1'][i]), doc_vec(train_data['question2'][i])))
        if prediction == train_data['is_duplicate'][i]: correct += 1.0

    return correct / len(train_data['question1'])

def main():
    global train_data

    print "Loading Question Data..."
    train_data = pandas.read_csv('data/train.csv')
    questions = map(lambda x: str(x), list(train_data['question1'].values) + list(train_data['question2'].values))

    print "Building vocabulary..."
    build_vocab(' '.join(questions))
    print "Vocabulary size: " + str(len(VOCAB))

    print "Accuracy: " + str(evaluate())

if __name__ == "__main__":
    main()
