import torch
import random
import time
import numpy as np
from common.corpus import getTaptapData
from common.logger import logger

class LstmCat(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, cat_dim, loss, embed, vocab, unk, layers=1, bidirectional=False):
        super(LstmCat, self).__init__()

        self.vocab = vocab 
        self.rvocab = {word: idx for idx, word in enumerate(self.vocab)}
        self.embed = embed
        self.UNK = unk
        self.bidirectional = bidirectional

        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional)
        if self.bidirectional:
            self.fc = torch.nn.Linear(hidden_dim * 2, cat_dim)
        else:
            self.fc = torch.nn.Linear(hidden_dim, cat_dim)
        self.loss = loss
        self.cat_dim = cat_dim

    def select(self, words):
        return self.embed.t()[words]

    def find(self, word):
        return self.rvocab.get(word, self.rvocab.get(self.UNK))

    def indexfy(self, words):
        _map = torch.tensor([self.find(word) for idx, word in enumerate(words)])
        return self.select(_map)

    def forward(self, XY):
        sentence, y = XY
        Y = torch.LongTensor([y, ])
        X = self.indexfy(sentence)

        data = list(X)
        data = [item.unsqueeze(1).t() for item in data]
        data = torch.cat(data, dim=0).unsqueeze(1)
        o, (hidden, cel) = self.lstm(data)
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        output = self.fc(hidden.squeeze(0)).sigmoid()
        if self.bidirectional:
            output = output.unsqueeze(0)
        loss = self.loss(output, Y)
        return output, loss

    def evaluate(self, test):
        correct = 0
        for XY in test:
            x, y = XY
            output, _ = self.forward(XY)
            result = torch.argmax(output)
            if result.item() == y:
                correct += 1
        return correct

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

def do_train(lstmcat, opti, XY, batch_size=50, epoches=5, report_every=2000):
    L = len(XY)
    ratio = 0.8
    split = int(ratio * L)
    train, test = XY[: split], XY[split: ]
    train = padData(train)
    trainL = len(train)
    mark = report_every
    for e in range(epoches):
        random.shuffle(train)
        process = 0
        loss_sum = 0

        for idx in range(0, len(train), batch_size):
            loss = 0
            batch = train[idx: idx + batch_size]
            l = len(batch)
            for XY in batch:
                process += 1
                _, _loss = lstmcat.forward(XY)
                loss += _loss
            loss = loss / batch_size

            loss_sum += loss.item() / l
            opti.zero_grad()
            loss.backward()
            opti.step()

            if not process > mark:
                logger.info("average loss: %f"%loss_sum)
                logger.info("training process: %d / %d"%(process, trainL))
                loss_sum = 0
                mark += report_every
        correct = lstmcat.evaluate(test)
        print("Epoch %d: %d / %d"%(e + 1, correct, len(test)))

if __name__ == '__main__':
    pass