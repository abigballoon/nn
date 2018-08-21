import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import random

torch.set_default_dtype(torch.float)

with open('vocab.dev', 'rb') as f:
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
        self.bed = torch.randn(self.embed_dim, self.vocab_dim, requires_grad=True)

    def find(self, word):
        try:
            idx = self.vocab.index(word)
        except ValueError:
            idx = len(self.vocab) - 1
        return idx

    def select(self, idx):
        _filter = torch.zeros(self.vocab_dim, 1)
        _filter[idx][0] = 1
        return self.bed.mm(_filter)

    def lookup(self, word):
        idx = self.find(word)
        return self.select(idx)

    def _forward(self, corpus, window=2, neg_size=5):
        for idx, word in enumerate(corpus):
            print(word)
            targets_start = idx - window
            if targets_start < 0:
                targets_start = 0
            targets = corpus[targets_start: idx] + corpus[idx + 1: idx + 1 + window]
            context_embeded = self.lookup(word)
            context_idx = self.find(word)
            loss_sum = torch.zeros(1, )
            for target in targets:
                target_embeded = self.lookup(target)
                loss = torch.log(torch.sigmoid(target_embeded.t().mm(context_embeded)))
                for _ in range(neg_size):
                    neg_embeded = self.negative(context_idx)
                    loss += torch.log(torch.sigmoid(-neg_embeded.t().mm(context_embeded)))
                if not torch.isnan(loss.sum()):
                    loss_sum += loss.sum()
            self.backward(loss_sum.sum(), len(targets))

    def forward(self, corpus, epochs, window=2, neg_size=5):
        for _ in range(epochs):
            self._forward(corpus, window=window, neg_size=neg_size)

    def backward(self, loss, batch_size):
        try:
            loss.backward()
        except RuntimeError:
            self.bed.grad.zero_()
            return
        with torch.no_grad():
            self.bed -= self.eta * self.bed.grad / batch_size
            self.bed.grad.zero_()

    def negative(self, idx):
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = random.randint(0, self.vocab_dim - 1)
        return self.select(neg_idx)


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
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=250, method="exact")
        low_dim_embs = tsne.fit_transform(self.bed.detach().t()[:plot_size, :])
        plot_with_labels(low_dim_embs, labels, "tsne.png")

model = Word2Vec(vocab, 100, eta=0.01)

model.forward(corpus, 1)
print("training done")

model.plot()