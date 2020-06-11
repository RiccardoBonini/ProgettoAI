import numpy as np

def kernel(x, y, d):
    return (1 + np.dot(x, y)) ** d

class KernelVotedPerceptron:
    def __init__(self, T, d):
        self.T = T
        self.C = []
        self.k = 0
        self.WY = []
        self.WX = []
        self.d = d
        self.errors = []

    def fit(self, x, y):

        k = 0
        wy = []
        wx = []
        c = [0]
        er = []
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
                    wy.append(y[i])
                    wx.append(x[i])
                    c.append(1)
                    k += 1
                if i % (len(x)/20) == 0: er.append(k)
                
        self.C = c
        self.k = k
        self.WY = wy
        self.WX = wx
        self.errors = er

    def vote(self, x):
        predictions = []
        for l in range(len(x)):
            s = 0
            for i in range(self.k):
                vx = 0
                for j in range(i):
                    vx = vx + self.WY[j] * kernel(self.WX[j], x[l], d = self.d)
                s = s + self.C[i] * np.sign(vx)
            predictions.append(np.sign(s))
        return predictions


