import os
import pickle
import random
import codecs
import re

from common.logger import logger

def getVocab(corpus):
    """
    @return [word, count][]
    """
    vocab = {}
    for line in corpus:
        for item in line:
            if item not in vocab:
                vocab[item] = 0
            vocab[item] += 1
    array = list(vocab.items())
    array.sort(key=lambda x: -x[1])
    return array

def sliceVocab(vocab, length, unk):
    logger.info("vocab length: %d"%len(vocab))
    minus = length - 1
    usable = vocab[: minus]
    dump = vocab[minus: ]
    usable.append((unk, sum([item[1] for item in dump])))
    return usable

def getIMDBData():
    if os.path.exists('imdb.corpus'):
        with open('imdb.corpus', 'rb') as f:
            data = pickle.load(f)
            return data

    def process(string):
        string = string.replace('\n', '')
        string = string.replace('<br />', ' ')
        for c in '()"\'<>,.':
            string = string.replace(c, ' '+c+' ')
        return string

    FP = './aclImdb/train/unsup/'
    fps = os.listdir(FP)
    random.shuffle(fps)
    lines = []
    for fp in fps:
        with codecs.open(os.path.join(FP, fp), encoding='utf8') as f:
            content = f.read()
        content = process(content)
        lines.append([item.strip().lower() for item in content.split(' ') if item.strip()])
    with open('imdb.corpus', 'wb+') as f:
        pickle.dump(lines, f)
    return lines

def getTaptapData():
    FP = 'yys.taptap.txt'
    with codecs.open(FP, encoding='utf8') as f:
        content = f.read()
    lines = content.split('\n')
    result = []
    for data in lines:
        line = data.replace('\\n', ' ').split('\t')[0]
        line = re.sub(' +', ' ', line)
        if not line: continue
        array = list(line)
        result.append(array)
    return result
