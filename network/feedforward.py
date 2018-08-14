import random
import torch
from dataset.mnist import data

data = list(data)
random.shuffle(data)
count = len(data)
split = int(count * 0.9)
train, test = data[: split], data[-(count - split): ]
torch.set_default_dtype(torch.float)

class NN(object):
    def __init__(self, shape, eta=None, device=None):
        self._dev = device or torch.device("cpu")
        self._eta = eta or 1e-3
        self._weights = []
        self._bias = []
        for d, next_d in zip(shape[: -1], shape[1: ]):
            weight = torch.randn(next_d, d, device=self._dev, dtype=torch.float, requires_grad=True)
            self._weights.append(weight)
            bias = torch.randn(next_d, 1, device=self._dev, dtype=torch.float, requires_grad=True)
            self._bias.append(bias)

    def _forward(self, data):
        a = data
        for w, b in zip(self._weights, self._bias):
            z = w.mm(a) + b
            a = torch.sigmoid(z)
        return a

    def forward(self, train, epochs, batch_size, test=None):
        for i in range(epochs):
            random.shuffle(train)
            print("Epoch %d starts"%(i+1))
            for index in range(0, len(train), batch_size):
                batch = train[index: index + batch_size]
                loss = torch.zeros(10, 1)
                for x, y in batch:
                    yhat = self._forward(x)
                    loss += (yhat - y).pow(2)
                loss_sum = loss.sum() / (2 * batch_size)
                self.backward(loss_sum, batch_size)
            if test:
                result = self.evaluate(test)
                print("Epoch %d: %d / %d"%(i+1, result, len(train)))
            else:
                print("Epoch %d ends"%(i+1))

    def backward(self, loss, batch_size):
        loss.backward()
        with torch.no_grad():
            for idx, w in enumerate(self._weights):
                w -= self._eta * w.grad / batch_size
                w.grad.zero_()
                self._weights[idx] = w

            for idx, b in enumerate(self._bias):
                b -= self._eta * b.grad / batch_size
                b.grad.zero_()
                self._bias[idx] = b

    def evaluate(self, test_data):
        test_results = []
        for x, y in test_data:
            yhat = torch.argmax(self._forward(x))
            test_results.append((yhat, torch.argmax(y)))
        return sum(int(x == y) for x, y in test_results)

# device = torch.device("cuda:0")
nn = NN((784, 30, 10), eta=1.2)
# x, y = train[0]
# print(y)
# print(x)
# result = nn._forward(x)
# print(result)
nn.forward(train, 30, 10, test)
