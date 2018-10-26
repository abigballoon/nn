# -*- coding: utf8 -*-

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import codecs
import numpy as np
import random

torch.set_default_dtype(torch.float64)

class NCE(torch.nn.Module):
    def __init__(self, vocab, embed_dim, unk):
        """
        vocab (string, number), word and its frequency
        """
        super(NCE, self).__init__()
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.embed = torch.nn.Parameter(torch.randn(self.embed_dim, self.vocab_size).uniform_(-1, 1))
        self.bias = torch.nn.Parameter(torch.randn(self.vocab_size, 1).uniform_(-1, 1))

        self.vocab = [item[0] for item in vocab]
        self.rvocab = {item: index for index, item in enumerate(self.vocab)}

        _sum = sum([item[1] ** 0.75 for item in vocab])
        self.freq = torch.tensor([item[1] ** 0.75 / _sum for item in vocab])
        self.UNK = unk

    def p(self, words):
        return self.freq[torch.argmax(words, 0)]
    
    def select(self, words):
        return self.embed.t()[words]

    def find(self, word):
        return self.rvocab.get(word, self.rvocab.get(self.UNK))

    def indexfy(self, words):
        """
        return (V, x), x = batch_size
        """
        _map = torch.tensor([[idx, self.find(word), ] for idx, word in enumerate(words)])
        return torch.sparse.FloatTensor(_map.t(), torch.ones(len(words)), torch.Size([len(words), self.vocab_size])).to_dense().t()

    def forward(self, targets, contexts, batch_size, noise_count):
        """
        target (V, x), one-hots, x = batch_size
        context (V, x), one-hots, x = batch_size
        self.embed (E, V)
        self.bias (V, 1)
        """
        # target rep
        q = torch.mm(self.embed, targets)

        # context rep
        r = torch.mm(self.embed, contexts)

        # bias
        b = torch.mm(self.bias.t(), targets)

        # score
        s = ((r * q).sum(0) + b) / self.embed_dim

        # distribution of words
        # p = self.p(targets + )

        loss = (s - torch.log(noise_count * self.p(targets))).sigmoid().log().sum()

        noises = self.get_noises(batch_size * noise_count)
        noise_loss = (1 - (torch.cat((s, ) * noise_count, dim=1) - torch.log(noise_count * self.p(noises))).sigmoid()).log().sum()

        return -(loss + noise_loss) / batch_size + 0.1 * self.embed.pow(2).sum() / (self.embed_dim * self.vocab_size)

    def get_targets_n_contexts(self, target_words, context_words):
        return self.indexfy(target_words), self.indexfy(context_words)

    def get_noises(self, count):
        choice = np.random.choice(self.vocab, count, p=self.freq)
        return self.indexfy(choice)

    def store(self):
        data = self.embed.detach()
        torch.save(data, 'model.torch')

    def load(self):
        data = torch.nn.Parameter(torch.load('model.torch'))
        self.embed = data


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

        print("start plot")
        labels = [self.vocab[i] for i in range(plot_size)]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method="exact")
        low_dim_embs = tsne.fit_transform(self.embed.detach().t()[:plot_size, :])
        plot_with_labels(low_dim_embs, labels, "tsne_nce.png")
        print("end plot")



def getData():
    with codecs.open("shakespear.dev.txt", encoding="utf8") as f:
        content = f.read().replace('\n', '').lower()
    return content.split(' ')

def getVocab(corpus):
    vocab = {}
    for item in corpus:
        if item not in vocab:
            vocab[item] = 0
        vocab[item] += 1
    array = list(vocab.items())
    array.sort(key=lambda x: -x[1])
    return array

def sliceVocab(vocab, length, unk):
    minus = length - 1
    usable = vocab[: minus]
    dump = vocab[minus: ]
    usable.append((unk, sum([item[1] for item in dump])))
    return usable

def getPair(corpus, window, batch_size):
    L = len(corpus)
    target = []
    context = []
    ids = [_ for _ in range(L)]
    random.shuffle(ids)
    for process, index in enumerate(ids):
        word = corpus[index]
        if not process % 5000:
            print(process / L)
        start = max(0, index - window)
        end = min(L, index + window + 1)
        for _context in corpus[start: index] + corpus[index + 1: end]:
            target.append(word)
            context.append(_context)
            if len(target) == batch_size:
                yield target, context
                target = []
                context = []
    yield target, context

if __name__ == '__main__':
    unk = '<UNK>'
    batch_size = 20
    neg_size = 5
    embed_dim = 300

    corpus = getData()
    print(len(corpus))
    vocab = getVocab(corpus)
    vocab = sliceVocab(vocab, 5000, unk)
    nce = NCE(vocab, embed_dim, unk)

    # nce.load()
    # nce.plot(200)
    # 1/0
    optimizer = torch.optim.SGD(nce.parameters(), 0.1)
    for _ in range(1):
        for targetwords, contextwords in getPair(corpus, 2, batch_size):
            targets, contexts = nce.get_targets_n_contexts(targetwords, contextwords)
            loss = nce.forward(targets, contexts, batch_size, neg_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # nce.store()
    # nce.plot()
