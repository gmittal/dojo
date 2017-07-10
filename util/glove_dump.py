# Dumps the model contents of a Glove text file into an easily accessible JSON file

import json, sys

def load_glove(path):
    print "Loading Glove Model"
    f = open(path,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model

def main():
    model = load_glove(sys.argv[1])
    with open('.'.join(sys.argv[1].split(".")[0:3] + ['json']), 'w') as out:
        json.dump(model, out)

if __name__ == '__main__':
    main()
