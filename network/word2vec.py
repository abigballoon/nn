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
    def __init__(self, vocab, freq, embed_dim, eta=None):
        self.eta = eta or 1.0
        self.vocab = vocab
        self.dict = {word: idx for idx, word in enumerate(vocab)}
        self.freq = freq
        self.max_freq = max(freq.values())
        self.vocab_dim = len(self.vocab)
        self.embed_dim = embed_dim
        self.ibed = torch.randn(self.embed_dim, self.vocab_dim, requires_grad=True)
        self.ibed_t = self.ibed.t()
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
        return self.ibed_t[ids]


    def lookup(self, words):
        ids = self.find(words)
        return self.select(ids)

    def gen(self, corpus, window=2, batch_size=5):
        L = len(corpus)
        result = []
        for idx in range(0, L):
            context_word = corpus[idx]
            targets_start = idx - window
            if targets_start < 0:
                targets_start = 0
            _targets = corpus[targets_start: idx] + corpus[idx + 1: idx + 1 + window]

            freq = float(self.freq.get(context_word, self.freq.get('<UNK>')))
            repeat = min(int(round(self.max_freq / freq)), 10)
            for _ in range(repeat):
                context_words = [context_word for _ in _targets]
                result.append([context_words, _targets, ])
                if len(result) == batch_size:
                    yield result
                    result = []


    def _forward(self, batch_size, corpus, window=2, neg_size=5):
        starttime = time.time()
        L = len(corpus)
        idx = 0
        for batch in self.gen(corpus, window, batch_size):
            if not idx % 1000:
                current = time.time()
                print("%f%%"%(idx / len(corpus) * 100))
                print("estimated:", (current - starttime) * (len(corpus) - idx) / 1000.0 / 60.0 / 60.0)
                starttime = time.time()
            idx += 1

            targets = []
            contexts1 = []
            contexts2 = []
            for context, target in batch:
                contexts1.extend(context)
                targets.extend(target)
                contexts2.extend(context * neg_size)
            target_embededs = self.lookup(targets)
            context_embededs = self.lookup(contexts1 + contexts2)
            neg_embededs = -self.negative(len(contexts2))

            a = torch.cat([target_embededs, neg_embededs], 0)
            loss = torch.log(torch.sigmoid(a * context_embededs))
            loss_sum = -loss.sum()
            if not torch.isnan(loss_sum) and not torch.isinf(loss_sum):
                self.backward(loss_sum, len(targets))

    def forward(self, batch_size, corpus, epochs, window=2, neg_size=5):
        for _ in range(epochs):
            self._forward(batch_size, corpus, window=window, neg_size=neg_size)

    def backward(self, loss, batch_size):
        loss.backward()
        with torch.no_grad():
            self.ibed -= (self.eta / batch_size) * self.ibed.grad
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

model = Word2Vec(vocab, vocabd, 300, eta=0.5)

model.forward(10, corpus, 3, neg_size=5)
print("training done")
model.store()
# model.load()
model.plot(700)