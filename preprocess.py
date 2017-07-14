# -*- coding: utf-8 -*-

import re
from util import contractions
from util import spelling

def words(text): return re.findall(r'\w+', text.lower())

def fix(text):
    expanded = contractions.expand(text)
    tokens = [spelling.correct(word) for word in words(expanded)]
    return ' '.join(tokens)
