import numpy as np
# from numba import vectorize, cuda
#
# @vectorize(['int32(int32, int32, int32)'], target = 'cuda')
def kernel(x, y, d):
    return (1 + np.dot(x, y)) ** d

class KernelVotedPerceptron:
    def __init__(self, T, d):
        self.T = T
        self.V = []
        self.C = []
        self.k = 0
        self.WY = []
        self.WX = []
        self.d = d

    def fit(self, x, y):

        k = 0
        v = [np.zeros_like(x[0])]
        wy = []
        wx = []
        c = [0]
        for epoch in range(self.T):
            for i in range(len(x)):
                s = 0
                if k == 0:
                    s = 0
                else:
                    for j in range(k):
                        s += wy[j] * kernel(wx[j], x[i], d = self.d)
                pred = 1 if s > 0 else -1
                if pred == y[i]:
                    c[k] += 1
                else:
                    v.append(np.add(v[k], np.dot(y[i], x[i])))
                    wy.append(y[i])
                    wx.append(x[i])
                    c.append(1)
                    k += 1

        self.V = v
        self.C = c
        self.k = k
        self.WY = wy
        self.WX = wx

    def getV(self):
        return self.V

    def getC(self):
        return self.C

    def getWY(self):
        return self.WY

    def getWX(self):
        return self.WX

    def getK(self):
        return self.k

    def predict(self, X):
        predictions = []
        for x in X:
            s = 0
            for i in range(self.k):
                s = s + self.C[i] * np.sign(np.dot(self.V[i], x))
            predictions.append(np.sign(1 if s > 0 else -1))
        return predictions
