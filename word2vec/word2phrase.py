# -*- coding: utf-8 -*-
import sys
import os
base = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, base)
from common.logger import logger

import collections
import codecs
import time
import pickle

class Word2Phrase(object):
    def __init__(self, text_file, threshold=50, dump_interval=20000, min_count=5, accept_count=200):
        self._fp = text_file
        self._text = ""
        self._counter = None
        self._phrases = []
        self._threshold = threshold
        self._last_dump = 0
        self._interval = dump_interval
        self._min_count = min_count
        self._accept_count = accept_count

    @property
    def _text_length(self):
        return len(self._text)

    def readfile(self):
        with codecs.open(self._fp, encoding="utf-8") as f:
            self._text = f.read()
        logger.info("text length: %d"%self._text_length)
        self._counter = dict(collections.Counter(self._text))
    
    def setText(self, text):
        self._text = text
        self._counter = dict(collections.Counter(self._text))

    def _interval_dump(self, index):
        if index - self._last_dump > self._interval:
            self.dump("self_build.dict")
            logger.info("current: %4f%%"%(float(index) * 100 / self._text_length))
            self._last_dump = index

    def _decide_phrase(self, prev, current):
        prev_count = self._counter[prev]
        current_count = self._counter[current]
        together = prev + current
        together_count = self._text.count(together)
        if together_count < self._min_count:
            return False, together_count

        prob = float(together_count) / (prev_count * current_count) * self._text_length
        threshold = self._threshold * (1 + 0.1 * (len(together) - 2))
        return prob > threshold, together_count

    def genPhrase(self):
        if not self._text:
            self._phrases = []
            return

        first_word = self._text[0]
        self._phrases.append(first_word)

        index = 1
        while index < self._text_length:
            self._interval_dump(index)
            prev_word = self._phrases[-1]
            current_word = self._text[index]

            isphrase, together_count = self._decide_phrase(prev_word, current_word)
            if isphrase:
                together = prev_word + current_word
                self._phrases[-1] = together
                self._counter[together] = together_count
            else:
                self._phrases.append(current_word)
            index += 1

    def dump(self, fp):
        with open(fp, "wb+") as f:
            pickle.dump(self._phrases, f)

obj = Word2Phrase("news.dev.txt")
obj.readfile()
obj.genPhrase()