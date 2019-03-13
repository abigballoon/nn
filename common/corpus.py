# -*- coding: utf8 -*-

import os
import pickle
import random
import codecs
import re
import numpy as np
import pkuseg

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

def segments(data, filename, nthread=20):
    """
    单线程分词太慢了，最好用并行分
    @input data: [sentence, label][]
    @input filename: 会用来存3个文件，
                     一个是等待分词的txt文件，
                     一个是segments的txt文件，
                     一个segments+label的pickle
    @return void, 会写一个pickle文件
    """
    corpus_fp = '%s.corpus.txt'%filename
    segments_fp = '%s.splited.txt'%filename
    pickle_fp = '%s.corpus.pickle'%filename
    with codecs.open(corpus_fp, 'w+', encoding='utf8') as f:
        for sentence, label in data:
            f.write(sentence)
            f.write('\n')
    pkuseg.test(corpus_fp, segments_fp, nthread=nthread)
    with codecs.open(segments_fp, encoding='utf8') as f:
        content = f.read()
    result = []
    for line, ori in zip(content.split('\n'), data):
        if not line: continue
        result.append((line.replace('\r', '').split(' '), ori[1], ))
    with open(pickle_fp, 'wb+') as f:
        pickle.dump(result, f)

def getTaptapData(fp, labeled=False):
    with codecs.open(fp, encoding='utf8') as f:
        content = f.read()
    lines = content.replace('\r\n', '\n').split('\n')
    result = []
    for data in lines:
        splited = data.replace('\\n', ' ').split('\t')
        if len(splited) != 2: continue
        line, cat = splited
        line = re.sub(' +', ' ', line)
        if not line: continue
        if not int(cat): continue
        if labeled:
            result.append((line, int(cat) - 1, ))
        else:
            result.append(line)
    return result

def splitTrainTest(XY, pad=True):
    L = len(XY)
    ratio = 0.8
    split = int(ratio * L)
    train, test = XY[: split], XY[split: ]
    if pad:
        train = padData(train)
    return train, test

def padData(data):
    result = {}
    max = 0
    for x, y in data:
        if y not in result:
            result[y] = []
        result[y].append(x)
        L = len(result[y])
        if L > max:
            max = L
    output = []
    for y, x in result.items():
        for item in x:
            output.append([item, y])
        for item in np.random.choice(x, max - len(x)):
            output.append([item, y])
    random.shuffle(output)
    return output
