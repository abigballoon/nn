# -*- coding: utf8 -*-
import torch
from torch import nn
import codecs
import time
import random
import string
import os

torch.set_default_dtype(torch.float)

SOF = "<S>"
UNK = "<UNK>"
EOF = "<E>"
vocab = [SOF, ] + list('abcdefghijklmnopqrstuvwxyz') + [EOF, UNK, ]
# vocab = list(string.ascii_letters) + list(" .,;'-") + [EOF, UNK, ]
print(vocab)
rvocab = {item: idx for idx, item in enumerate(vocab)}
vocab_size = len(vocab)
h_size = 256
print(vocab_size)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0005):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell_state_size = self.input_size + self.hidden_size

        # forget gate
        self.i2forget = nn.Linear(input_size + hidden_size, hidden_size)
        # input gate
        self.i2input =  nn.Linear(input_size + hidden_size, hidden_size)
        self.i2cell = nn.Linear(input_size + hidden_size, hidden_size)

        # output gate
        self.i2output = nn.Linear(input_size + hidden_size, hidden_size)

        # h2o
        self.h2o = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.learning_rate = learning_rate


    def forward(self, i, h, c):
        """
        i (1, input_size)
        h (1, hidden_size)
        c (1, hidden_size)

        ot (1, hidden_size) 
        ht (1, hidden_size)
        ct (1, hidden_size)
        ft (1, hidden_size)
        it (1, hidden_size)
        """

        input_combined = torch.cat([i, h], dim=1)
        # forget gate
        ft = self.i2forget(input_combined).sigmoid()
        it = self.i2input(input_combined).sigmoid()

        ctbar = self.i2cell(input_combined).tanh()
        ct = ft * c + it * ctbar

        ot = self.i2output(input_combined).sigmoid()
        ht = ot * ct.tanh()

        output = self.h2o(ht)
        output = self.dropout(output)
        output = self.softmax(output)

        return it, ht, ct, output

    def predict(self, pre):
        result = pre
        h = self.initState()
        c = self.initState()
        X, _ = vecs(pre, rvocab, SOF, EOF, UNK)
        predict_c = None
        loop = 0
        while predict_c != EOF and len(result) < 20:
            if loop < len(pre):
                i = X[loop]
            i, h, c, o = self.forward(i.unsqueeze(0), h, c)
            c_i = torch.argmax(o)
            predict_c = vocab[c_i.item()]
            result = result + [predict_c, ]
            _X, _ = vecs([predict_c, ], rvocab, SOF, EOF, UNK)
            i = _X[0]
            loop += 1
        return ''.join(result)

    def initState(self):
        return torch.zeros(1, self.hidden_size)

    def backward(self):
        for p in self.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)

def vecs(word, rvocab, sof, eof, unk):
    word = [sof, ] + list(word) + [eof, ]
    def find(character):
        return rvocab.get(character, rvocab.get(unk))

    _map = torch.tensor([[idx, find(item)] for idx, item in enumerate(word)])
    L = len(word)
    word_vec = torch.sparse.FloatTensor(_map.t(), torch.ones(L), torch.Size([L, len(rvocab)])).to_dense()
    return word_vec[: -1], torch.LongTensor([find(item) for item in word[1: ]])

def getENData():
    with codecs.open("en.dev.txt", encoding="utf8") as f:
        content = f.read().replace('\n', '').lower()
    words = list(filter(lambda x: len(x) > 1, content.split(' ')))
    result = []
    for item in words:
        if item not in result:
            result.append(item)
    return result

def getNamesData():
    path = '/home/cgz/Downloads/names/names/'
    names = []
    fps = os.listdir(path)
    fps = ['English.txt', 'Scottish.txt']
    for txt in fps:
        with codecs.open(os.path.join(path, txt), encoding="utf8") as f:
            content = f.read()
            names += list(filter(lambda x: len(x) > 1, content.split('\n')))
    return names

def train(lstm):
    if not lstm:
        lstm = LSTM(vocab_size, h_size, vocab_size)

    words = getENData()
    random.shuffle(words)
    L = len(words)
    print(L)
    sum_loss = 0
    for process, word in enumerate(words):
        X, Y = vecs(word, rvocab, SOF, EOF, UNK)
        h = lstm.initState()
        c = lstm.initState()
        loss = 0
        lstm.zero_grad()
        for idx in range(X.size()[0]):
            """
            x (input_size, )
            y (1, )
            """
            x = X[idx]
            y = Y[idx]

            i, h, c, o = lstm(x.unsqueeze(0), h, c)
            l = lstm.loss(o, y.unsqueeze(0))

            loss += l
            sum_loss += loss

        loss.backward()
        lstm.backward()

        if not (process + 1) % 3000:
            print(word)
            print(sum_loss / 3000)
            print((process + 1) / L)
            sum_loss = 0
    return lstm

if __name__ == '__main__':
    lstm = LSTM(vocab_size, h_size, vocab_size)
    # lstm = None
    for _ in range(20):
        lstm = train(lstm)
        torch.save(lstm, "lstm.torch")
        starts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'u', 'x']
        for c in starts:
            print(lstm.predict([sof, c]))

    starts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'u', 'x']
    for c in starts:
        print(lstm.predict([sof, c]))

