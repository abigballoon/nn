import torch
from torch import nn
import codecs
import time
import random
import string

torch.set_default_dtype(torch.float)

SOF = "<S>"
UNK = "<UNK>"
EOF = "<E>"
# vocab = list('abcdefghijklmnopqrstuvwxyz') + [EOF, UNK, ]
vocab = [SOF, ] + list(string.ascii_letters) + [EOF, UNK, ]
rvocab = {item: idx for idx, item in enumerate(vocab)}
vocab_size = len(vocab)
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

        self.dropout = nn.Dropout(0.1)
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
        out_combined = torch.cat([o, next_h], dim=1)
        output = self.o2o(out_combined)
        drop = self.dropout(output)
        final_output = self.softmax(drop)
        return final_output.t(), next_h

    def backward(self):
        for p in self.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)

        self.zero_grad()

    def predict(self, pre):
        result = pre
        h = torch.zeros(1, self.hidden_size)
        X, _ = vecs(pre, rvocab, SOF, EOF, UNK)
        predict_c = None
        loop = 0
        while predict_c != EOF and len(result) < 20:
            if loop < len(pre):
                i = X[loop]
            o, _ = self.forward(i.unsqueeze(0).t(), h.t())
            c_i = torch.argmax(o)
            predict_c = vocab[c_i.item()]
            result = result + [predict_c, ]
            i = o.t().squeeze(0)
            loop += 1
        return ''.join(result)

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
        content = f.read().replace('\n', '')
    return list(filter(lambda x: len(x) > 1, content.split(' ')))

def getNamesData():
    with codecs.open("/home/cgz/Downloads/names/names/English.txt", encoding="utf8") as f:
        content = f.read()
    return list(filter(lambda x: len(x) > 1, content.split('\n')))

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
        for idx in range(X.size()[0]):
            x = X[idx]
            y = Y[idx]

            o, h = rnn.forward(x.unsqueeze(0).t(), h.t())
            l = rnn.loss(o.t(), y.unsqueeze(0))
            loss += l
            sum_loss += loss
        loss.backward()
        rnn.backward()

        if not (process + 1) % 5000:
            print(word)
            print(sum_loss / 5000)
            print((process + 1) / L)
            sum_loss = 0
    return rnn
    

if __name__ == '__main__':
    # rnn = torch.load("rnn.torch")
    rnn = None
    for _ in range(2):
        rnn = train(rnn)
        torch.save(rnn, "rnn.torch")
    print(rnn.predict([SOF, "Hel"]))
    print(rnn.predict([SOF, "He"]))
    print(rnn.predict([SOF, "H"]))
    print(rnn.predict([SOF, 'Th']))
    print(rnn.predict([SOF, 'A']))
    print(rnn.predict([SOF, 'B']))
    print(rnn.predict([SOF, 'C']))
    print(rnn.predict([SOF, 'D']))
    print(rnn.predict([SOF, 'E']))
    print(rnn.predict([SOF, 'F']))

