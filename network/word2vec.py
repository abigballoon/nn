import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import random
import time
import json
import codecs

torch.set_default_dtype(torch.float64)

def sample_from_corpus(vocab, corpus, window):
    rdict = {word: idx for idx, word in enumerate(vocab)}
    def find(word):
        try:
            return rdict[word]
        except KeyError:
            return -1

    def gen():
        for idx, word in enumerate(corpus):
            targets_start = idx - window
            if targets_start < 0:
                targets_start = 0
            _targets = corpus[targets_start: idx] + corpus[idx + 1: idx + 1 + window]
            for target in _targets:
                yield find(word), find(target)

    array = list(gen())
    random.shuffle(array)
    return array


def strip(s):
    items = ',.\'!?(); '
    for c in items:
        s = s.replace(c, '')
    return s.lower()

with codecs.open('shakespear.dev.txt', 'r', encoding='utf8') as f:
    corpus = f.read()
    corpus = corpus.replace('\n', ' <CR> ')
    corpus = [strip(item) for item in corpus.split(' ') if strip(item)]
print('corpus:', len(corpus))
vocabd = {}
for word in corpus:
    if word not in vocabd:
        vocabd[word] = 0
    vocabd[word] += 1
vocab = list(vocabd.items())
vocab.sort(key=lambda x: x[1], reverse=True)
vocab.append("<UNK>")
print(vocab[: 10])
vocab = [item[0] for item in vocab[: 49999]]
print('vocab:', len(vocab))

samples = sample_from_corpus(vocab, corpus, 2)
print(samples[:10])

class Word2Vec(object):
    def __init__(self, vocab, freq, embed_dim, eta=None):
        self.eta = eta or 1.0
        self.vocab = vocab
        self.dict = {word: idx for idx, word in enumerate(vocab)}
        self.max_freq = max(freq.values())
        self.vocab_dim = len(self.vocab)
        self.embed_dim = embed_dim
        self.ibed = torch.randn(self.embed_dim, self.vocab_dim, requires_grad=True)
        self.ibed_t = self.ibed.t()


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

    def gen(self, samples, window=2, batch_size=5):
        L = len(samples)
        contexts, targets = [], []
        random.shuffle(samples)
        for idx in range(0, L, batch_size):
            for context, target in samples[idx: idx + batch_size]:
                contexts.append(context)
                targets.append(target)
            yield idx, contexts, targets
            contexts, targets = [], []

    def _forward(self, batch_size, samples, window=2, neg_size=5):
        starttime = time.time()
        L = len(corpus)
        prev = 0
        total_loss = 0
        for idx, contexts, targets in self.gen(samples, window, batch_size):
            if idx - prev > 1000:
                current = time.time()
                print("%f%%"%(idx / len(corpus) * 100))
                print("estimated:", (current - starttime) * (len(corpus) - idx) / float(idx - prev) / 60.0 / 60.0)
                print("average loss:", total_loss / float(idx - prev))
                starttime = time.time()
                prev = idx
                total_loss = 0

            target_embededs = self.select(targets)
            context_embededs = self.select(contexts * (neg_size + 1))
            neg_embededs = self.negative(len(contexts) * neg_size)

            a = torch.cat([target_embededs, neg_embededs], 0)
            loss = torch.log(torch.sigmoid(a * context_embededs))
            loss_sum = -loss.sum()
            total_loss += loss_sum.item()
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

    def negative(self, count=5):
        return -self.select([random.randint(0, self.vocab_dim - 1) for _ in range(count)])


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
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method="exact")
        low_dim_embs = tsne.fit_transform(self.ibed.detach().t()[:plot_size, :])
        plot_with_labels(low_dim_embs, labels, "tsne.png")

model = Word2Vec(vocab, vocabd, 300, eta=0.1)

model.forward(100, samples, 1, neg_size=10)
print("training done")
model.store()
# model.load()
model.plot(300)