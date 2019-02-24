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
vocab = list('abcdefghijklmnopqrstuvwxyz') + [EOF, UNK, ]
# vocab = list(string.ascii_letters) + list(" .,;'-") + [EOF, UNK, ]
print(vocab)
rvocab = {item: idx for idx, item in enumerate(vocab)}
vocab_size = len(vocab)
h_size = 256
print(vocab_size)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0005):
        """
        input (input_size, 1)
        hidden (hidden_size, input_size)
        output (output_size, hidden_size)
        """
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)

        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.learning_rate = learning_rate

    def forward(self, i, h):
        """
        i (input_size, 1)
        h (hidden_size, 1)
        o (input_size, )

        in_combined (input_size + hidden_size, 1)
        """
        in_combined = torch.cat([i, h])
        o = self.i2o(in_combined.t())
        next_h = self.i2h(in_combined.t())
        out_combined = torch.cat([next_h, o], dim=1)
        output = self.o2o(out_combined)
        drop = self.dropout(output)
        final_output = self.softmax(drop)
        return final_output.t(), next_h

    def backward(self):
        for p in self.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)

    def predict(self, pre):
        result = pre
        h = torch.zeros(1, self.hidden_size)
        X, _ = vecs(pre, rvocab, SOF, EOF, UNK)
        predict_c = None
        loop = 0
        while predict_c != EOF and len(result) < 20:
            if loop < len(pre):
                i = X[loop]
            o, h = self.forward(i.unsqueeze(0).t(), h.t())
            c_i = torch.argmax(o)
            predict_c = vocab[c_i.item()]
            result = result + [predict_c, ]
            _X, _ = vecs([predict_c, ], rvocab, SOF, EOF, UNK)
            i = _X[0]
            loop += 1
        return ''.join(result)

def vecs(word, rvocab, sof, eof, unk):
    word = list(word) + [eof, ]
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

def train(rnn=None):
    h_size = 200
    if not rnn:
        rnn = RNN(vocab_size, h_size, vocab_size)

    words = getENData()
    random.shuffle(words)
    L = len(words)
    print(L)
    sum_loss = 0
    for process, word in enumerate(words):
        X, Y = vecs(word, rvocab, SOF, EOF, UNK)
        h = torch.zeros(1, h_size)
        loss = 0
        rnn.zero_grad()
        for idx in range(X.size()[0]):
            x = X[idx]
            y = Y[idx]

            o, h = rnn(x.unsqueeze(0).t(), h.t())
            l = rnn.loss(o.t(), y.unsqueeze(0))

            loss += l
            sum_loss += loss

        loss.backward()
        rnn.backward()

        if not (process + 1) % 3000:
            print(word)
            print(sum_loss / 3000)
            print((process + 1) / L)
            sum_loss = 0
    return rnn
    

if __name__ == '__main__':
    # rnn = torch.load("rnn.torch")
    rnn = None
    for _ in range(20):
        rnn = train(rnn)
        torch.save(rnn, "rnn.torch")
        starts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'u', 'x']
        for c in starts:
            print(rnn.predict([c]))

