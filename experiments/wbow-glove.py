# -*- coding: utf-8 -*-

# Quora Duplicate Question Evaluation Task
# Weighted Bag of Words Model with Glove-based Word Vector Representations to Generate Sentence Embeddings

import numpy as np
import spacy, pandas, tqdm, sys

reload(sys)
sys.setdefaultencoding('utf-8')
nlp = spacy.load('en')

glove = {}
word_probs = {}
direction = []

def load_glove_vecs(path='glove/glove.840B.300d.json'):
    return json.loads(open(path).read())

def get_word_probs(sent_list):
    total = 0.0
    VOCAB = {}
    for i in tqdm.tqdm(range(len(sent_list))):
        sentence = sent_list[i]
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
    try:
        return a / (a + word_probs[word.lower()])
    except KeyError:
        return 1

def sent_vec(sent):
    doc = nlp(unicode(sent))
    vec = np.zeros(300)
    for word in doc: vec += word_weight(word.text) * np.asarray(glove[word.text.lower()])
    return vec / len(doc)

def cos_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def evaluate():
    correct = 0.0
    for i in tqdm.tqdm(range(len(train_data['question1']))):
        prediction = round(cos_sim(sent_vec(train_data['question1'][i]), sent_vec(train_data['question2'][i])))
        if prediction == train_data['is_duplicate'][i]: correct += 1.0
    return correct / len(train_data['question1'])

def init():
    global glove, word_probs

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

    print "Accuracy: " + str(evaluate())


if __name__ == "__main__":
    init()
