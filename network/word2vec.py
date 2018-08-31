import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import random
import time
import json

torch.set_default_dtype(torch.float64)

# with open('shakespeare.dev', 'rb') as f:
with open('shakespeare.dev', 'rb') as f:
    corpus = pickle.load(f)

vocabd = {}
for word in corpus:
    if word not in vocabd:
        vocabd[word] = 0
    vocabd[word] += 1
print(len(corpus))
vocab = list(vocabd.items())
vocab.sort(key=lambda x: x[1], reverse=True)

vocab = [item[0] for item in vocab[: 4999]]
vocab.append("<UNK>")

class Word2Vec(object):
    def __init__(self, vocab, embed_dim, eta=None):
        self.eta = eta or 1.0
        self.vocab = vocab
        self.dict = {word: idx for idx, word in enumerate(vocab)}
        self.vocab_dim = len(self.vocab)
        self.embed_dim = embed_dim
        self.ibed = torch.randn(self.embed_dim, self.vocab_dim, requires_grad=True)
        self.obed = torch.randn(self.embed_dim, self.vocab_dim, requires_grad=True)

    def find(self, words):
        ids = []
        for word in words:
            idx = self.dict.get(word, None)
            if idx == None:
                idx = self.dict.get('<UNK>')
            ids.append(idx)
        return ids

    def select(self, ids, word_type='i'):
        return self.ibed.t()[ids]


    def lookup(self, words):
        ids = self.find(words)
        return self.select(ids)

    def _forward(self, batch_size, corpus, window=2, neg_size=5):
        starttime = time.time()
        L = len(corpus)
        for start in range(0, L, batch_size):
            end = start + batch_size
            if end > L:
                end = L
            targets = []
            contexts1 = []
            contexts2 = []
            for idx in range(start, end):
                if not idx % 1000:
                    current = time.time()
                    print("%f%%"%(idx / len(corpus) * 100))
                    print("estimated:", (current - starttime) * (len(corpus) - idx) / 1000.0 / 60.0 / 60.0)
                    starttime = time.time()

                targets_start = idx - window
                if targets_start < 0:
                    targets_start = 0
                _targets = corpus[targets_start: idx] + corpus[idx + 1: idx + 1 + window]
                targets += _targets
                contexts1 += [idx for _ in _targets]
                contexts2 += [idx for _ in range(neg_size * len(_targets))]
            target_embededs = self.lookup(targets)
            context_embededs = self.lookup(contexts1)
            loss = torch.log(torch.sigmoid(target_embededs * context_embededs)).sum()
            neg_embededs = self.negative(len(contexts2))
            context_embededs = self.lookup(contexts2)
            loss += torch.log(torch.sigmoid(-neg_embededs * context_embededs)).sum()

            if not torch.isnan(loss.sum()) and not torch.isinf(loss.sum()):
                loss_sum = -loss.sum()
                self.backward(loss_sum.sum(), len(targets))

        """
        for idx, word in enumerate(corpus):
            if not idx % 1000:
                current = time.time()
                print("%f%%"%(idx / len(corpus) * 100))
                print("estimated:", (current - start) * (len(corpus) - idx) / 1000.0 / 60.0 / 60.0)
                start = time.time()
            if random.random() > 0.3:
                continue
            targets_start = idx - window
            if targets_start < 0:
                targets_start = 0
            targets = corpus[targets_start: idx] + corpus[idx + 1: idx + 1 + window]
            context_embededs = self.lookup([idx for _ in targets], 'i')
            loss_sum = torch.zeros(1, )
            switch = False

            target_embededs = self.lookup(targets, 'o')
            loss = torch.log(torch.sigmoid(target_embededs * context_embededs)).sum()

            neg_embededs = self.negative(neg_size)
            context_embededs = self.lookup([idx for _ in range(neg_size)], 'i')
            loss += torch.log(torch.sigmoid(-neg_embededs * context_embededs)).sum()
            if not torch.isnan(loss.sum()) and not torch.isinf(loss.sum()):
                loss_sum -= loss.sum()
                switch = True
            if switch:
                self.backward(loss_sum.sum(), len(targets))
        """

    def forward(self, batch_size, corpus, epochs, window=2, neg_size=5):
        for _ in range(epochs):
            self._forward(batch_size, corpus, window=window, neg_size=neg_size)

    def backward(self, loss, batch_size):
        loss.backward()
        with torch.no_grad():
            self.ibed -= self.eta * self.ibed.grad / batch_size
            self.ibed.grad.zero_()
            # self.obed -= self.eta * self.obed.grad / batch_size
            # self.obed.grad.zero_()

    def negative(self, count=5):
        return torch.randn(count, self.embed_dim)


    def store(self):
        data = self.ibed.detach()
        torch.save(data, 'model.torch')

    def load(self):
        data = torch.load('model.torch')
        self.ibed = data

    def plot(self, plot_size=500):
        def plot_with_labels(low_dim_embs, labels, filename):
            plt.rcParams['font.sans-serif'] = ['SimHei', ]
            plt.rcParams['axes.unicode_minus'] = False
            plt.figure(figsize=(20, 20))
            for i, label in enumerate(labels):
                x, y = low_dim_embs[i, :]
                plt.scatter(x, y)
                plt.annotate(
                    label,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords="offset points",
                    ha="right",
                    va="bottom"
                )
            plt.savefig(filename)

        labels = [self.vocab[i] for i in range(plot_size)]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=7000, method="exact")
        low_dim_embs = tsne.fit_transform(self.ibed.detach().t()[:plot_size, :])
        plot_with_labels(low_dim_embs, labels, "tsne.png")

model = Word2Vec(vocab, 300, eta=0.5)

# model.forward(20, corpus, 3, neg_size=5)
print("training done")
# model.store()
model.load()
model.plot(700)