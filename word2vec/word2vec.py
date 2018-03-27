# -*- encoding = utf8 -*-
"""
参照tensorflow教程写的
https://www.tensorflow.org/tutorials/word2vec
"""
import sys
import os
base = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, base)
from common.logger import logger


import pickle
import math
import codecs
import collections
import random

logger.info("import module")
import jieba
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def getData():
    with open("text.dev", "br") as f:
        vocab = pickle.load(f)
    return vocab

class Word2Vec(object):
    def __init__(self, vocab_fp, train_fp, embedding_size=300, batch_size=128, vocab_size=50000, eta=0.1, window_size=3):
        self._vocab_fp = vocab_fp
        self._train_fp = train_fp
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.eta = eta
        self.window_size = window_size
        self._final_embeddings = None
        self._load()
        self._embed()


    def _load(self):
        with codecs.open(self._vocab_fp, "br") as f:
            self.vocab_all = pickle.load(f)
        logger.info("loaded vocab size: %d"%len(self.vocab_all))

        counts = [["UNK", -1]]
        counts.extend(collections.Counter(self.vocab_all).most_common(self.vocab_size - 1))
        logger.info("counts len: %d"%len(counts))

        dictionary = dict()
        for word, _ in counts:
            dictionary[word] = len(dictionary)

        logger.info("creating mapping")
        data = list()
        unk_count = 0
        for word in self.vocab_all:
            index = dictionary.get(word, 0)
            if not index:
                unk_count += 1
            data.append(index)
        counts[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        logger.info("mapping complete")
        logger.info("dictionary size: %d"%len(dictionary))
        self.dict = dictionary
        self.rdict = reversed_dictionary
        self.vocab_counts = counts

        with codecs.open(self._train_fp, encoding="utf8") as f:
            content = f.read()
            self.train_segments = jieba.cut(content, cut_all=False)
            self.train_content = content.split('\n')

    def _embed(self):
        with tf.name_scope("embedding"):
            vocab_size = self.vocab_size
            embedding_size = self.embedding_size
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
            )
            self.embeddings = embeddings

    def _init_weights_and_bias(self):
        vocab_size = self.vocab_size
        embedding_size = self.embedding_size
        with tf.name_scope("weights"):
            self.nce_weights = tf.Variable(
                tf.truncated_normal([vocab_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size))
            )
        with tf.name_scope("biases"):
            self.nce_bias = tf.Variable(tf.zeros([vocab_size]))
            # self.nce_bias = tf.Variable(
            #     tf.truncated_normal([vocab_size, ],
            #                         stddev=1.0 / math.sqrt(vocab_size))
            # )
            pass

    def _pre_train(self):
        num_sampled = 64
        with tf.device('/cpu:0'):
            self._init_weights_and_bias()
            with tf.name_scope("inputs"):
                self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            with tf.name_scope("embed"):
                self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

        self._sess = tf.Session()
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                            biases=self.nce_bias,
                            labels=self.train_labels,
                            inputs=self.embed,
                            num_sampled=num_sampled,
                            num_classes=self.vocab_size)
            )
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.eta).minimize(self.loss)

    def get_center_words(self, c_index):
        pass

    def get_next_pair(self, skip_window):

        def generate_from(array):
            center_index = int((len(array) + 1) / 2)
            center = array[center_index]
            for item in array[: center_index] + array[center_index+1: ]:
                yield center, item

        stop = False
        window = []
        while not stop:
            try:
                window.append(self.train_segments.__next__())
            except StopIteration:
                stop = True

            if len(window) != skip_window * 2 + 1:
                continue
            
            for _input, label in generate_from(window):
                yield _input, label
            window = window[1:]

    def generate_batch(self, batch_size, skip_window):
        inputs, labels = [], []
        for _input, label in self.get_next_pair(skip_window):
            inputs.append(self.dict.get(_input, 0))
            labels.append(self.dict.get(label, 0))
            if len(inputs) == batch_size:
                yield inputs, labels
                inputs, labels = [], []

    def gen_segments(self, content):
        return jieba.cut('\n'.join(content), cut_all=False)

    def _gen_segments(self):
        self.train_segments = self.gen_segments(self.train_content)

    def train_one_epoch(self, num_sampled=64, learning_rate=1.0):
        random.shuffle(self.train_content)
        self._gen_segments()
        overall_loss = 0
        count = 0
        for inputs, labels in self.generate_batch(self.batch_size, self.window_size):
            inputs = np.array(inputs).reshape((len(inputs), ))
            labels = np.array(labels).reshape((len(labels), 1))
            feed = {self.train_inputs: inputs, self.train_labels: labels}
            _, loss = self._sess.run([self.optimizer, self.loss], feed)
            overall_loss += loss
            count += 1
        logger.info("epoch average loss: %s"%str(overall_loss / float(count)))

    def train(self, epoch):
        self._pre_train()
        init = tf.global_variables_initializer()
        init.run(session=self._sess)
        for i in range(epoch):
            logger.info("%d epoch started"%i)
            self.train_one_epoch()
        logger.info("train finished")

    @property
    def final_embeddings(self):
        if self._final_embeddings is None:
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            normalized_embeddings = self.embeddings / norm
            self._final_embeddings = normalized_embeddings.eval(session=self._sess)
        return self._final_embeddings

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

        labels = [self.rdict[i] for i in range(plot_size)]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method="exact")
        low_dim_embs = tsne.fit_transform(self.final_embeddings[:plot_size, :])
        plot_with_labels(low_dim_embs, labels, "tsne.png")
    
    def distance(self, w1, w2):
        id1 = self.dict.get(w1, 0)
        id2 = self.dict.get(w2, 0)
        v1 = self.final_embeddings[id1]
        v2 = self.final_embeddings[id2]

        _sum = 0
        for f1, f2 in zip(v1, v2):
            _sum += (f1 - f2) ** 2
        return _sum ** 0.5
    
    def closest(self, target, count=10):
        result = []
        for _id, word in self.rdict.items():
            if word == target:
                continue
            result.append((_id, self.distance(target, word)))
        result.sort(key=lambda x: x[1])
        result = [self.rdict[item[0]] for item in result[: count]]
        return result

app = Word2Vec("vocab.dev", "news.dev.txt", window_size=2)
app.train(25)
# print(app.closest(u"中国"))
app.plot()
