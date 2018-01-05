# -*- coding: utf-8 -*-

import re, sys, pandas, tqdm
from util import contractions
from util import symspell

DICTIONARY = []

def words(text): return re.findall(r'\w+', text.lower())

def load_dictionary(path):
    return set(words(open(path).read()))

def OOV(word):
    return not word in DICTIONARY

def correct(word):
    try:
        return symspell.get_suggestions(word)[0] if OOV(word) else word
    except:
        return word

def fix(text):
    try:
        text = contractions.expand(text) if text.index("'") > -1 else text
    except:
        pass
    
    tokens = [correct(word) for word in words(text)]
    return ' '.join(tokens)

def init():
    global DICTIONARY
    DICTIONARY = load_dictionary('data/big.txt')
    symspell.create_dictionary('data/big.txt')
    
init()

if __name__ == "__main__":
    DATA = pandas.read_csv(sys.argv[1])
    with open(sys.argv[1] + '.preprocess', 'w') as out:
        out.write('id,question\n')
        for i in tqdm.tqdm(range(len(DATA['id']))):
            out.write(DATA['id'][i] + ',' + fix(DATA['question'][i]) + '\n')
        out.close()
