import numpy as np

from common.math import EMPTY

class Neuron(object):
    def __init__(self, input_size, weights=None, bias=None):
        if weights:
            if input_size != len(weights):
                raise ValueError("length of weights should equal to input_size")
        weights = weights or [0 for _ in range(input_size)]
        self.weights = np.array(weights)
        self.bias = bias or 0

        self._input = EMPTY
        self._output = EMPTY

    def input(self, data):
        self._input = data

    @property
    def output(self):
        if self._input == EMPTY:
            raise ValueError("no input yet")

        if self._output == EMPTY:
            self._output = self.process()
        return self._output

    def process(self):
        raise NotImplementedError

class Perceptron(Neuron):
    def __init__(self, input_size, threshold, **kwargs):
        super(Perceptron, self).__init__(input_size, **kwargs)
        self.threshold = threshold

    def process(self):
        return int(np.sum(self._input * self.weights + self.bias) > self.threshold)