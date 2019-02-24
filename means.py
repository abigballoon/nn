# -*- coding: utf8 -*-

import os
import codecs
import random
import pickle
import torch
from network.torchlstm import LSTM
from network.feedforward import NN
from network.nce import getVocab as getVocab2
import time

torch.set_default_dtype(torch.float32)

def process(string):
    string = string.replace('\n', '')
    string = string.replace('<br />', ' ')
    for c in '()"\'<>,.':
        string = string.replace(c, ' '+c+' ')
    return string

def _getXY(fp, y):
    fps = os.listdir(fp)
    result = []
    for item in fps:
        with codecs.open(os.path.join(fp, item), encoding='utf8') as f:
            content = f.read()
        result.append(([item.strip().lower() for item in process(content).split(' ') if item.strip()], y, ))
    return result

def getXY():
    pos = _getXY('./aclImdb/train/pos/', 1)
    neg = _getXY('./aclImdb/train/neg/', 0)
    data = pos + neg
    random.shuffle(data)
    return data

def getXYFrom(fp='imdb.train'):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    random.shuffle(data)
    return data

def getVocab(fp='imdb.vocab'):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data

def sliceVocab(vocab, length, unk):
    print(len(vocab))
    minus = length - 1
    usable = vocab[: minus]
    dump = vocab[minus: ]
    usable.append((unk, sum([item[1] for item in dump])))
    return usable

def loadPretrained(fp, words, dim):
    result = []
    with open(fp) as f:
        line = f.readline()
        while line:
            content = line.split(' ')
            word = content[0] 
            if word not in words:
                vec = [0 for _ in range(dim)]
            else:
                vec = [float(item) for item in content[1: ]]
            result.append(vec)
            if len(result) >= len(words):
                break
            line = f.readline()
    return torch.tensor(result).t()

class Fit(torch.nn.Module):
    def __init__(self, ):
        super(Fit, self).__init__()
        self.rawvocab = getVocab('rawvocab.pickle')
        self.UNK = '<UNK>'
        self.vocab = [item[0] for item in sliceVocab(self.rawvocab, 100000, self.UNK)]
        print(self.vocab[: 10])
        print(self.vocab[9470])
        self.rvocab = {word: idx for idx, word in enumerate(self.vocab)}

        self.embed = torch.load('imdb2.torch').float()
        # self.embed = loadPretrained('/home/cgz/Downloads/glove.6B/glove.6B.300d.txt', self.vocab, 300)
        self.lstm = torch.nn.LSTM(300, 256, num_layers=2, bidirectional=True)
        self.fc = torch.nn.Linear(256 * 2, 1)
        self.loss= torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.parameters(), 0.005)

    def select(self, words):
        return self.embed.t()[words]

    def find(self, word):
        return self.rvocab.get(word, self.rvocab.get(self.UNK))

    def indexfy(self, words):
        _map = torch.tensor([self.find(word) for idx, word in enumerate(words)])
        return self.select(_map)

    def _sum(self, data, sum_size, step):
        L = len(data)
        if L <= sum_size:
            step = L
        for idx in range(0, L, step):
            segment = data[idx: idx + sum_size]
            yield segment.mean(0)

    def closest(self, word, count=5):
        print(self.rvocab[word])
        vec = self.indexfy([word, ])
        cos = torch.nn.CosineSimilarity(dim=1)
        distances = cos(vec, self.embed.t())
        _, idx = distances.topk(count)
        for distance, item in zip(_, idx):
            print(self.vocab[item], distance.item())

    def evaluate(self, test, sum_size, overlap):
        correct = 0
        for sentence, y in test:
            # neg = 0; pos = 1
            X = self.indexfy(sentence)
            # h = self.lstm.initState()
            # c = self.lstm.initState()
            self.lstm.zero_grad()
            # for x in self._sum(X, sum_size, int(sum_size * (1 - overlap))):
            # data = list(self._sum(X, sum_size, int(sum_size * (1 - overlap))))
            # output = self.lstm(data)
            # data = list(self._sum(X, sum_size, int(sum_size * (1 - overlap))))
            data = list(X[: 100])
            data = [item.unsqueeze(1).t() for item in data]
            data = torch.cat(data, dim=0).unsqueeze(1)
            o, (hidden, cel) = self.lstm(data)
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
    
            # output = self.h2o(torch.cat([h, hb], dim=1))
            # output = self.o2o(output).sigmoid()
            # output = self.m(output)
            output = self.fc(hidden.squeeze(0))
 
            # idx = torch.argmax(o)
            # if idx == y:
            #     correct += 1
            o = output.sigmoid() 
            if (o.item() > 0.5 and y) :
                correct += 1
            if (o.item() < 0.5 and not y) :
                correct += 1
        return correct


    def trainnn(self, epochs):
        def build(data):
            result = []
            for x, y in data:
                item = torch.FloatTensor([[0], [0]])
                item[y][0] = 1
                result.append((
                    self.indexfy(x).mean(0).unsqueeze(0).t(),
                    # torch.FloatTensor([[y]])
                    item,
                ))
            return result

        XY = getXYFrom()
        random.shuffle(XY)
        print(len(XY))
        XY = build(XY)
        ratio = 0.8
        L = len(XY)
        split = int(ratio * L)
        train, test = XY[: split], XY[split: ]
        nn = NN((300, 100, 100, 2), eta=3)
        nn.forward(train, epochs, 50, test)

    def train(self, epochs, sum_size=10, overlap=0.5):
        XY = getXYFrom()
        ratio = 0.8
        L = len(XY)
        split = int(ratio * L)
        train, test = XY[: split], XY[split: ]
        for e in range(epochs):
            random.shuffle(train)
            process = 0
            loss_sum = 0
            stamp = time.time()

            for idx in range(0, len(train), 50):
                loss = 0
                self.zero_grad()
                for sentence, y in train[idx: idx+50]:
                    process += 1
                    # neg = 0; pos = 1
                    Y = torch.FloatTensor([y])
                    # Y = torch.LongTensor([y])
                    X = self.indexfy(sentence)
    
                    # for x in self._sum(X, sum_size, int(sum_size * (1 - overlap))):
                    # data = list(self._sum(X, sum_size, int(sum_size * (1 - overlap))))
                    data = list(X[: 100])
                    data = [item.unsqueeze(1).t() for item in data]
                    data = torch.cat(data, dim=0).unsqueeze(1)
                    o, (hidden, cel) = self.lstm(data)
                    hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
    
                    # output = self.h2o(torch.cat([h, hb], dim=1))
                    # output = self.o2o(output).sigmoid()
                    # output = self.m(output)
                    output = self.fc(hidden.squeeze(0))
                    loss += self.loss(output.sigmoid(), Y)

                loss_sum += loss.item()
                loss.backward()
                self.optim.step()
    
                if not process % 2000:
                    print(loss_sum)
                    print(process / len(train))
                    print((time.time() - stamp) / 60.0)
                    stamp = time.time()
                    loss_sum = 0
                    self.store()
            correct = self.evaluate(test, sum_size, overlap)
            print("Epoch %d: %d / %d"%(e + 1, correct, len(test)))

    def store(self):
        torch.save(self, 'means.torch')


if __name__ == '__main__':
    # data = getXY()
    # with open('imdb.train', 'wb+') as f:
    #     pickle.dump(data, f)
    # f = Fit()
    f = torch.load('means.torch')
    # f.trainnn(100)
    f.train(10, 20, 0.2)
    f.closest('king')
    f.closest('big')
    f.closest('good')
    f.closest('him')
    f.closest('don')
