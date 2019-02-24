import torch
import random

DTYPE = torch.float

class RNN(object):
    def __init__(self, shape, eta):
        self._input_dim = shape[0]
        self._hidden_states = []
        self._U = []
        self._V = []
        self._W = []
        self.eta = eta

        for hidden_dim in shape[1: ]:
            u = torch.randn(hidden_dim, self._input_dim, dtype=DTYPE, requires_grad=True)
            v = torch.randn(self._input_dim, hidden_dim, dtype=DTYPE, requires_grad=True)
            w = torch.randn(hidden_dim, hidden_dim, dtype=DTYPE, requires_grad=True)

            self._hidden_states.append(hidden_dim)
            self._U.append(u)
            self._V.append(v)
            self._W.append(w)

    def _forward(self, X):
        def layer(h, u, v, w):
            T = len(X)
            s = []
            yhat = []
            for _ in range(T + 1):
                s.append(torch.zeros(h, 1, dtype=DTYPE, requires_grad=True))
            for _ in range(T):
                yhat.append(torch.zeros(self._input_dim, 1, requires_grad=True))

            for t, x in enumerate(X):
                s[t] = torch.tanh(u.mm(x) + w.mm(s[t-1]))
                yhat[t] = torch.nn.functional.softmax(v.mm(s[t]), 0)
            return yhat, s

        for idx, hidden_dim in enumerate(self._hidden_states):
            u, v, w = self._U[idx], self._V[idx], self._W[idx]
            X, s = layer(hidden_dim, u, v, w)
        return X, s

    def predict(self, X):
        y, s = self._forward(X)
        return y

    def forward(self, train, epochs, batch_size, eta=None, test=None):
        """
        train: Array of (1, x) shape matrix
        """
        for i in range(epochs):
            random.shuffle(train)
            train_size = len(train)
            for index in range(0, train_size, batch_size):
                batch = train[index: index+batch_size]
                loss_sum = torch.zeros(1, )
                for X, Y in batch:
                    yhat, s = self._forward(X)
                    for yhatt, yt in zip(yhat, Y):
                        # loss_sum += -yt.mm(torch.log(yhatt)).sum() / len(Y)
                        tmp = (-yt * torch.log(yhatt) - (1 - yt) * torch.log(1 - yhatt)).sum()
                        if not torch.isnan(tmp).item() and not torch.isinf(tmp).item():
                            loss_sum += tmp
                loss_sum = loss_sum / batch_size
                self.backward(loss_sum, batch_size)
            print(loss_sum)

    def backward(self, loss, batch_size):
        loss.backward()
        with torch.no_grad():
            for u in self._U:
                u -= self.eta * u.grad / batch_size
                u.grad.zero_()
            for v in self._V:
                v -= self.eta * v.grad / batch_size
                v.grad.zero_()
            for w in self._W:
                w -= self.eta * w.grad / batch_size
                w.grad.zero_()


if __name__ == '__main__':
    nn = RNN((5, 10), eta=0.5)
    nn.forward(train, 30, 10)
