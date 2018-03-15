# -*- encoding = utf8 -*-
"""
参照tensorflow教程写的
https://www.tensorflow.org/tutorials/word2vec
"""

import pickle
import math
# 要install个jieba
import tensorflow as tf

def getData():
    pass

class Word2Vec(object):
    def __init__(self, vocab_fp, train_fp, embedding_size=300, batch_size=10):
        self._vocab_fp = vocab_fp
        self._train_fp = train_fp
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self._set_default()

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _set_default(self):
        self.vocab = []
        self.train_content = ""

    def _load(self):
        with codecs.open(self._vocab_fp, encoding="utf8") as f:
            self.vocab = pickle.load(f)

        with codecs.open(self._train_fp, encoding="utf8") as f:
            self.train_content = f.read()

    def _embed(self):
        vocab_size = self.vocab_size
        embedding_size = self.embedding_size
        embeddings = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
        )

    def _init_weights_and_bias(self):
        vocab_size = self.vocab_size
        embedding_size = self.embedding_size
        self.nce_weights = tf.Variable(
            tf.truncated_normal([vocab_size, embedding_size]
                                stddev=1.0 / math.sqrt(embedding_size))
        )
        self.nce_bias = tf.Variable(tf.zeros([vocab_size]))

    def _pre_train(self):
        self._train()
        self._init_weights_and_bias()
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        self.embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        self._sess = tf.Session()
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.nce_weights,
                        biases=self.nce_bias
                        labels=self.train_labels,
                        inputs=self.embed,
                        num_sampled=num_sampled,
                        num_classes=self.vocab_size)
        )
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    def get_center_words(self, c_index):
        pass


    def generate_batch(self):
        pass

    def train(self, num_sampled=64, learning_rate=1.0):
        for inputs, labels in self.generate_batch():
            feed = {"train_inputs": inputs, "train_labels": labels}
            self._sess.run([self.optimizer, self.loss], feed)