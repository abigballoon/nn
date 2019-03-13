import pickle

def loadPickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)