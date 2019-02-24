
import random

from common.math import sigmoid, sigmoid_prime, mse_derivative
from common.logger import logger

def _r(width):
    return list(np.random.randn(width))
    sign = 1 if random.random() > 0.5 else -1
    return [random.random() * sign for _ in range(width)]


class network(object):
    def __init__(self, shape, activation_function=sigmoid):
        _weights = []
        _bias = []
        self._input_size = shape[0]
        self._layers = len(shape)
        prev_layer_size = self._input_size
        for layer_size in shape[1: ]:
            layer_weight = np.array([_r(layer_size) for _ in range(prev_layer_size)])
            _weights.append(layer_weight)
            layer_bias = np.array(_r(layer_size))
            _bias.append(layer_bias)
            prev_layer_size = layer_size
        self._weights = np.array(_weights)
        self._bias = np.array(_bias)

        self._activation_function = activation_function

    def feed(self, data):
        if len(data) != self._input_size:
            raise TypeError("excepted (%d, ) data, got %s"%
                            (self._input_size, str(np.array(data).shape)))
        data = np.array(data)

        for index, weight in enumerate(self._weights):
            data = np.array([self._activation_function(item)\
                            for item in (data.dot(weight) + self._bias[index])])

        return data

    def learn(self, data, epochs, mini_batch_size, eta, test_data):
        data_size = len(data)
        for _ in range(epochs):
            logger.info("starting epchos #%d"%_)
            random.shuffle(data)
            mini_batches = self.get_mini_batches(data, mini_batch_size)
            nabla_w, nabla_b = None, None

            for batch in mini_batches:
                batch_nabla_w, batch_nabla_b = self.learn_from_batch(batch)
                # if nabla_w == None or not nabla_b == None:
                #     nabla_w = [eta * item / float(data_size) for item in batch_nabla_w]
                #     nabla_b = [eta * item / float(data_size) for item in batch_nabla_b]
                # else:
                #     nabla_w = add(nabla_w, [eta * item / float(data_size) for item in batch_nabla_w])
                #     nabla_b = add(nabla_b, [eta * item / float(data_size) for item in batch_nabla_b])

                nabla_w = [eta * item / float(len(batch)) for item in batch_nabla_w]
                nabla_b = [eta * item / float(len(batch)) for item in batch_nabla_b]
                self._weights = minus(self._weights, nabla_w)
                self._bias = minus(self._bias, nabla_b)

            # test
            total_test = len(test_data)
            correct = self.evaluate(test_data)
            logger.info("epchos #%d test result: %d / %d"%(_, correct, total_test))

    def learn_from_batch(self, data):
        nabla_w_sum, nabla_b_sum = None, None
        for x, y in data:
            y = np.array(y)
            nabla_w = [None for _ in range(self._layers - 1)]
            nabla_b = [None for _ in range(self._layers - 1)]
            zs = []
            activations = [np.array(x).reshape(1, len(x)), ]
            activation = np.array(x).reshape(1, len(x))
            for index, weight in enumerate(self._weights):
                z = activation.dot(weight) + self._bias[index]
                activation = np.array([self._activation_function(item) for item in z])
                zs.append(z)
                activations.append(activation)

            error = mse_derivative(activation, y.reshape(1, len(y))) * sigmoid_prime(z)
            nabla_b[-1] = error
            nabla_w[-1] = activations[-2].transpose().dot(error)
            error_minus_one= None
            for l in range(2, self._layers):
                error = self._weights[-l+1].dot(error.transpose()).transpose() * sigmoid_prime(zs[-l])
                nabla_b[-l] = error
                nabla_w[-l] = activations[-l-1].transpose().dot(error)

            if nabla_w_sum == None or nabla_b_sum == None:
                nabla_w_sum, nabla_b_sum = nabla_w, nabla_b
            else:
                nabla_w_sum = add(nabla_w_sum, nabla_w)
                nabla_b_sum = add(nabla_b_sum, nabla_b)

        return nabla_w_sum, nabla_b_sum

    def get_mini_batches(self, data, size):
        for index in range(0, len(data), size):
            yield data[index: index + size]

    def evaluate(self, test_data):
        correct = 0
        for x, y in test_data:
            predict_index = np.argmax(self.feed(x))
            if y[predict_index]:
                correct += 1
        return correct


def add(array1, array2):
    result = []
    for item1, item2 in zip(array1, array2):
        result.append(item1 + item2)
    return result

def minus(array1, array2):
    result = []
    for item1, item2 in zip(array1, array2):
        result.append(item1 - item2)
    return result

def equal(array1, array2):
    if len(array1) != len(array2):
        raise Exception("should have equal length")
    for x, y in zip(array1, array2):
        if x != y:
            return False
    return True
