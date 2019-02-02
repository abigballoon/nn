# -*- coding: utf8 -*-

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import codecs
import numpy as np
import random
import pickle
import os
import time
from matplotlib.font_manager import FontProperties

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
        self._vocab = list(reversed(self.vocab))
        self.rvocab = {item: index for index, item in enumerate(self.vocab)}

        _sum = sum([item[1] ** 0.75 for item in vocab])
        self.freq = torch.tensor([item[1] ** 0.75 / _sum for item in vocab])
        self.UNK = unk

    def p(self, words):
        return self.freq[words]
    
    def select(self, words):
        return self.embed.t()[words]

    def find(self, word):
        return self.rvocab.get(word, self.rvocab.get(self.UNK))

    def indexfy(self, words):
        """
        return (V, x), x = batch_size
        """
        _map = torch.tensor([self.find(word) for idx, word in enumerate(words)])
        return _map

    def forward(self, targets, contexts, noise_count, re=10):
        """
        target (V, x), one-hots, x = batch_size
        context (V, x), one-hots, x = batch_size
        target (x, ), indexs, x = batch_size
        context (x, ), indexs, x = batch_size
        self.embed (E, V)
        self.bias (V, 1)
        """
        batch_size = targets.size()[0]
        # target rep
        q = self.embed.t()[targets].t()

        # context rep
        r = self.embed.t()[contexts].t()

        # bias
        b = self.bias[targets].t()

        # score
        s = ((r * q).sum(0) + b) / self.embed_dim

        loss = (s - torch.log(noise_count * self.p(targets))).sigmoid().log().sum()

        noises = self.get_noises(batch_size * noise_count)

        noise_loss = (1 - (torch.cat((s, ) * noise_count, dim=1) - torch.log(noise_count * self.p(noises))).sigmoid()).log().sum()

        total_loss = -(loss + noise_loss) / batch_size
        penalty = re * (q.pow(2).sum() + r.pow(2).sum()) / (self.embed_dim * batch_size)
        # print(total_loss)
        # print(penalty)
        del q
        del r
        del b
        del s
        del loss
        del noises
        del noise_loss

        return total_loss + penalty

    def get_targets_n_contexts(self, target_words, context_words):
        return self.indexfy(target_words), self.indexfy(context_words)

    def get_noises(self, count):
        choice = np.random.choice(self.vocab, count)
        result = self.indexfy(choice)
        return result

    def store(self, name=None):
        torch.save(self, 'nce.torch')

        with open('vocab.pickle', 'wb+') as f:
            pickle.dump(self.vocab, f)

    def dump(self, name=None):
        data = self.embed.detach()
        torch.save(data, 'imdb2.torch')

    def load(self):
        return
        data = torch.nn.Parameter(torch.load('model.torch'))
        self.embed = data
        print(self.embed)


def plot(self, plot_size=500):
    def plot_with_labels(low_dim_embs, labels, filename):
        font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
        # plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', ]
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(40, 40))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontproperties=font,
            )
        plt.savefig(filename)

    print("start plot")
    labels = [self.vocab[i] for i in range(plot_size)]
    data = self.embed.detach().t()[:plot_size, :]
    del self
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method="exact")
    low_dim_embs = tsne.fit_transform(data)
    plot_with_labels(low_dim_embs, labels, "tsne_nce.png")
    print("end plot")



def getCNData():
    with codecs.open("news.dev.txt", encoding="utf8") as f:
        content = f.read().replace(' ', '').lower()
    return list(content)

def getENData():
    with codecs.open("en.dev.txt", encoding="utf8") as f:
        content = f.read().replace('\n', '').lower()
    return content.split(' ')

def getIMDBData():
    if os.path.exists('imdb.corpus'):
        with open('imdb.corpus', 'rb') as f:
            data = pickle.load(f)
            return data

    def process(string):
        string = string.replace('\n', '')
        string = string.replace('<br />', ' ')
        for c in '()"\'<>,.':
            string = string.replace(c, ' '+c+' ')
        return string

    FP = './aclImdb/train/unsup/'
    fps = os.listdir(FP)
    random.shuffle(fps)
    lines = []
    for fp in fps:
        with codecs.open(os.path.join(FP, fp), encoding='utf8') as f:
            content = f.read()
        content = process(content)
        lines.append([item.strip().lower() for item in content.split(' ') if item.strip()])
    with open('imdb.corpus', 'wb+') as f:
        pickle.dump(lines, f)
    return lines

def getVocab(corpus):
    vocab = {}
    for line in corpus:
        for item in line:
            if item not in vocab:
                vocab[item] = 0
            vocab[item] += 1
    array = list(vocab.items())
    array.sort(key=lambda x: -x[1])
    with open('rawvocab.pickle', 'wb+') as f:
        pickle.dump(array, f)
    return array

def sliceVocab(vocab, length, unk):
    print(len(vocab))
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
        if not process % 7000:
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

def _getPair(corpus, window):
    """
    corpus Array<string>
    """
    L = len(corpus)
    target = []
    context = []
    ids = [_ for _ in range(L)]
    # random.shuffle(ids)
    for process, index in enumerate(ids):
        word = corpus[index]
        start = max(0, index - window)
        end = min(L, index + window + 1)
        for _context in corpus[start: index] + corpus[index + 1: end]:
            yield word, _context

def getLinePair(corpus, window, batch_size):
    """
    corpus Array<Array<string>>
    """
    data = []
    for line in corpus:
        data = list(_getPair(line, window))
        random.shuffle(data)
        for index in range(0, len(data), batch_size):
            batch = data[index: index + batch_size]
            yield [item[0] for item in batch], [item[1] for item in batch]


if __name__ == '__main__':
    unk = '<UNK>'
    batch_size = 600
    neg_size = 60
    embed_dim = 300

    corpus = getIMDBData()
    print('corpus length', len(corpus))
    # vocab = getVocab(corpus)
    with open('rawvocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
        print('vocob length', len(vocab))
    vocab = sliceVocab(vocab, 100000, unk)
    # nce = NCE(vocab, embed_dim, unk)
    del vocab
    nce = torch.load('nce.torch')

    # nce.load()
    # plot(nce, 900)
    # nce.dump()
    # 1/0
    optimizer = torch.optim.SGD(nce.parameters(), 0.08)
    for _ in range(5):
        process = 0
        sum_loss = 0
        start = time.time()
        for targetwords, contextwords in getLinePair(corpus, 2, batch_size):
            targets, contexts = nce.get_targets_n_contexts(targetwords, contextwords)
            loss = nce.forward(targets, contexts, neg_size, 3.0)
            sum_loss += loss.item()
            if not process % 1000:
                print(nce.embed)
                print(sum_loss / 1000.0)
                sum_loss = 0
                start = time.time()
                nce.store()
                print("process saved")
            if torch.isinf(loss):
                print("got inf")
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # clean up
            del targets
            del contexts
            del loss

            process += 1
            if not process % 50:
                print("remain %0.2f"%(((time.time() - start) / process) * (1000 - process % 1000)))
    nce.store()
    nce.plot()
