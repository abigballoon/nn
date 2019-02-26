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
from common.logger import logger
from common.err import StorePathFileExist

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

        # loss = (s - torch.log(noise_count * self.p(targets))).sigmoid().log().sum()
        loss = (s).sigmoid().log().sum() / batch_size

        noises = self.get_noises(batch_size * noise_count)
        nt = torch.cat((targets, ) * noise_count, dim=0)
        # noise target rep
        nq = self.embed.t()[nt].t()

        # noise context rep
        nr = self.embed.t()[noises].t()

        # noise bias
        nb = self.bias[nt].t()

        # noise score
        ns = ((nr * nq).sum(0) + nb) / self.embed_dim

        # noise_loss = (1 - (torch.cat((s, ) * noise_count, dim=1) - torch.log(noise_count * self.p(noises))).sigmoid()).log().sum()
        # np = self.p(torch.cat((targets, ) * noise_count, dim=0))
        # noise_loss = (1 - (ns - torch.log(np)).sigmoid()).log().sum()# * noise_count
        noise_loss = (1 - (ns).sigmoid()).log().sum() / (batch_size * noise_count)
        # noise_loss = (1 - ns.sigmoid()).log().sum() / noise_count

        total_loss = -(loss + noise_loss)
        # penalty = re * (q.pow(2).sum() + r.pow(2).sum()) / (self.embed_dim * batch_size)
        # penalty += re * (nq.pow(2).sum() + nr.pow(2).sum()) / (self.embed_dim * batch_size * noise_count)
        del q
        del r
        del b
        del s
        del loss
        del noises
        del noise_loss
        return total_loss# + penalty

    def get_targets_n_contexts(self, target_words, context_words):
        return self.indexfy(target_words), self.indexfy(context_words)

    def get_noises(self, count):
        choice = np.random.choice(self.vocab, count, p=self.freq)
        # choice = np.random.choice(self.vocab, count)
        result = self.indexfy(choice)
        return result

    def dump(self, name=None):
        data = self.embed.detach()
        torch.save(data, 'imdb2.torch')

    def load(self):
        return
        data = torch.nn.Parameter(torch.load('model.torch'))
        self.embed = data


def plot(nce, fp, plot_size=500):
    def plot_with_labels(low_dim_embs, labels, filename):
        plt.rcParams['font.sans-serif'] = ['SimHei', ]
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
            )
        plt.savefig(filename)

    logger.info("start plot")
    labels = [nce.vocab[i] for i in range(plot_size)]
    data = nce.embed.detach().t()[:plot_size, :]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method="exact")
    low_dim_embs = tsne.fit_transform(data)
    plot_with_labels(low_dim_embs, labels, fp)
    logger.info("end plot")

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
    """
    @return [word, count][]
    """
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
    logger.info("corpus length: %d"%len(vocab))
    minus = length - 1
    usable = vocab[: minus]
    dump = vocab[minus: ]
    usable.append((unk, sum([item[1] for item in dump])))
    return usable

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
    corpus Array<Array<string>> """
    step = 100
    random.shuffle(corpus)
    for lineidx in range(0, len(corpus), step):
        data = []
        lines = corpus[lineidx: lineidx + step]
        for line in lines:
            data += list(_getPair(line, window))
        random.shuffle(data)
        for index in range(0, len(data), batch_size):
            batch = data[index: index + batch_size]
            yield [item[0] for item in batch], [item[1] for item in batch]

def check_file_exists(fp, overwrite):
    if not overwrite and os.path.exists(fp):
        raise StorePathFileExist(fp)

def default_data_saver(obj, fp):
    torch.save(obj, fp)
    logger.info("process saved")

def do_train(nce, optimizer, corpus, batch_size=60, neg_size=60, epoch=5, report_every=500, countdown_every=50, datasaver=None, save_fp=None, save_overwrite=False):
    """
    @input corpus: string[][], array of sentences, sentences are array of words.
    """
    check_file_exists(save_fp, save_overwrite)
    datasaver = datasaver or default_data_saver
    for _ in range(epoch):
        process = 0
        sum_loss = 0
        start = time.time()
        for targetwords, contextwords in getLinePair(corpus, 2, batch_size):
            targets, contexts = nce.get_targets_n_contexts(targetwords, contextwords)
            loss = nce.forward(targets, contexts, neg_size, 0)
            sum_loss += loss.item()
            if not process % report_every:
                logger.info(nce.embed)
                logger.info("average loss: %f"%(sum_loss / float(report_every)))
                sum_loss = 0
                start = time.time()
                default_data_saver(nce, save_fp)
            if torch.isinf(loss):
                logger.warning("got inf")
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # clean up
            del targets
            del contexts
            del loss

            process += 1
            if not process % countdown_every:
                real_process = process % report_every
                if real_process:
                    logger.info("remain %0.2fs"%(((time.time() - start) / real_process) * (report_every - real_process)))

if __name__ == '__main__':
    unk = '<UNK>'

    corpus = getIMDBData()
    logger.info('corpus length', len(corpus))
    # vocab = getVocab(corpus)
    with open('rawvocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
        logger.info('vocob length', len(vocab))
    vocab = sliceVocab(vocab, 100000, unk)
    # vocab = sliceVocab(vocab, 5000, unk)

    DUMP = False
    RELOAD = True
    if DUMP or RELOAD:
        nce = torch.load('nce.torch')
    else:
        nce = NCE(vocab, embed_dim, unk)
    # optimizer = torch.optim.SGD(nce.parameters(), 2.5)
    opti = torch.optim.Adam(nce.parameters())

    del vocab
    print(nce.embed)

    # nce.load()
    if DUMP:
        plot(nce, 900)
        nce.dump()
        1/0

    do_train(nce, optimizer=opti, corpus=corpus, save_fp='nce.torch', save_overwrite=True)
