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
        self._load()
        self._embed()


    def _load(self):
        with codecs.open(self._vocab_fp, "br") as f:
            self.vocab_all = pickle.load(f)
        logger.info("loaded vocab size: %d"%len(self.vocab_all))

        counts = [["UNK", -1]]
        counts.extend(collections.Counter(self.vocab_all).most_common(self.vocab_size - 1))

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
        self.dict = dictionary
        self.rdict = reversed_dictionary
        self.vocab_counts = counts

        with codecs.open(self._train_fp, encoding="utf8") as f:
            self.train_content = f.read().split('\n')

    def _embed(self):
        vocab_size = self.vocab_size
        embedding_size = self.embedding_size
        embeddings = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
        )
        self.embeddings = embeddings

    def _init_weights_and_bias(self):
        vocab_size = self.vocab_size
        embedding_size = self.embedding_size
        self.nce_weights = tf.Variable(
            tf.truncated_normal([vocab_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size))
        )
        self.nce_bias = tf.Variable(tf.zeros([vocab_size]))

    def _pre_train(self):
        num_sampled = 64
        self._init_weights_and_bias()
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
        self._sess = tf.Session()
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.nce_weights,
                        biases=self.nce_bias,
                        labels=self.train_labels,
                        inputs=self.embed,
                        num_sampled=num_sampled,
                        num_classes=self.vocab_size)
        )
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.eta).minimize(self.loss)

    def get_center_words(self, c_index):
        pass


    def generate_batch(self, batch_size, skip_window):
        for index in range(0, len(self.train_content), batch_size):
            logger.info("generating %dth batch"%index)
            inputs, labels = [], []
            segments = []
            for line in self.train_content[index: index + batch_size]:
                segments.extend(jieba.cut(line, cut_all=False))

            for sindex in range(len(segments)):
                center = segments[sindex]
                center_id = self.dict.get(center, 0)
                left = segments[index - 1 - skip_window: index - 1]
                right = segments[index + 1: index + 1 + skip_window]

                for context in left + right:
                    context_id = self.dict.get(context, 0)
                    inputs.append(center_id)
                    labels.append((context_id, ))
            yield inputs, labels


    def train(self, num_sampled=64, learning_rate=1.0):
        random.shuffle(self.train_content)
        self._pre_train()
        for inputs, labels in self.generate_batch(self.batch_size, self.window_size):
            feed = {self.train_inputs: inputs, self.train_labels: labels}
            _, summary, loss_val = self._sess.run([self.optimizer, self.loss], feed)

app = Word2Vec("vocab.dev", "news.dev.txt")
app.train()