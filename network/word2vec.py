import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import random

torch.set_default_dtype(torch.float64)

with open('shakespeare.dev', 'rb') as f:
    corpus = pickle.load(f)

vocabd = {}
for word in corpus:
    if word not in vocabd:
        vocabd[word] = 0
    vocabd[word] += 1
vocab = list(vocabd.items())
vocab.sort(key=lambda x: x[1], reverse=True)

vocab = [item[0] for item in vocab[: 4999]]
vocab.append("<UNK>")

class Word2Vec(object):
    def __init__(self, vocab, embed_dim, eta=None):
        self.eta = eta or 1.0
        self.vocab = vocab
        self.vocab_dim = len(self.vocab)
        self.embed_dim = embed_dim
        self.ibed = torch.randn(self.embed_dim, self.vocab_dim, requires_grad=True)
        self.obed = torch.randn(self.embed_dim, self.vocab_dim, requires_grad=True)

    def find(self, word):
        try:
            idx = self.vocab.index(word)
        except ValueError:
            idx = len(self.vocab) - 1
        return idx

    def select(self, idx, word_type='i'):
        _filter = torch.zeros(self.vocab_dim, 1)
        _filter[idx][0] = 1
        if word_type == 'i':
            return self.ibed.mm(_filter)
        else:
            return self.ibed.mm(_filter)

    def lookup(self, word, word_type):
        idx = self.find(word)
        return self.select(idx, word_type)

    def _forward(self, corpus, window=2, neg_size=5):
        for idx, word in enumerate(corpus):
            if not idx % 1000:
                print("%f%%"%(idx / len(corpus) * 100))
            targets_start = idx - window
            if targets_start < 0:
                targets_start = 0
            targets = corpus[targets_start: idx] + corpus[idx + 1: idx + 1 + window]
            context_embeded = self.lookup(word, 'i')
            context_idx = self.find(word)
            loss_sum = torch.zeros(1, )
            switch = False

            for target in targets:
                target_embeded = self.lookup(target, 'o')
                loss = torch.log(torch.sigmoid(target_embeded.t().mm(context_embeded)))
                for _ in range(neg_size):
                    neg_embeded = self.negative(context_idx)
                    loss += torch.log(torch.sigmoid(-neg_embeded.t().mm(context_embeded)))
                if not torch.isnan(loss.sum()) and not torch.isinf(loss.sum()):
                    loss_sum -= loss.sum()
                    switch = True
            if switch:
                self.backward(loss_sum.sum(), len(targets))

    def forward(self, corpus, epochs, window=2, neg_size=5):
        for _ in range(epochs):
            self._forward(corpus, window=window, neg_size=neg_size)

    def backward(self, loss, batch_size):
        loss.backward()
        print(loss.sum())
        with torch.no_grad():
            self.ibed -= self.eta * self.ibed.grad / batch_size
            self.ibed.grad.zero_()
            # self.obed -= self.eta * self.obed.grad / batch_size
            # self.obed.grad.zero_()

    def negative(self, idx):
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = random.randint(0, self.vocab_dim - 1)
        return self.select(neg_idx, 'o')


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

model = Word2Vec(vocab, 300, eta=0.5)

model.forward(corpus, 1)
print("training done")

model.plot()