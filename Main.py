import pandas as pd
import mnist
import numpy as np
import KernelVotedPerceptron as kvp
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import mnist_reader
from numpy import linalg as lng

#ds = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

# train_images = mnist.train_images()
# X = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
# X_train = X[0:2000]
# y = mnist.train_labels()
# y_train = y[0:2000]
#
# test_images = mnist.test_images()
# X = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
# X_test = X[0:500]
# y = mnist.test_labels()
# y_test = y[0:500]

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')



print(X_test[0])
print(y_test)

dimension = 2;

kernel_voted_perceptron = kvp.KernelVotedPerceptron(T=10, d = dimension)

score = [0,0,0,0,0,0,0,0,0,0]

weights = []
WY = []
WX = []
cost = []
k = []
y = []
z = []

for j in range(len(y_train)):
    y.append(0)


for j in range(len(y_test)):
    z.append(0)

#training kernel voted perceptron sulle 10 classi
for i in range(0,10):
 for n in range(len(y_train)):
     if y_train[n] == i:
         y[n] = 1
     else :
         y[n] = -1
 kernel_voted_perceptron.fit(X_train, y)
 weights.append(kernel_voted_perceptron.getV())
 WY.append(kernel_voted_perceptron.getWY())
 WX.append(kernel_voted_perceptron.getWX())
 cost.append(kernel_voted_perceptron.getC())
 k.append(kernel_voted_perceptron.getK())

#KERNEL LAST (UNNORMALIZED) METHOD
for n in range(len(X_test)):
    for i in range(0,10):

        score[i]=np.dot(weights[i][-1], X_test[n])

    z[n] = np.argmax(score)

print(z)
print('Accuracy score for unnormalized last method for KERNEL perceptron:', accuracy_score(y_test, z))
print("Classification report for unnormalized last method for KERNEL perceptron: %s:\n%s\n4", classification_report(y_test, z))


#KERNEL LAST (NORMALIZED) METHOD
for n in range(len(X_test)):
    for i in range(0,10):

        score[i]= np.dot(weights[i][-1], X_test[n]) / lng.norm(weights[i][-1], 2)

    z[n] = np.argmax(score)

print(z)
print('Accuracy score for normalized last method for KERNEL perceptron', accuracy_score(y_test, z))
print("Classification report for normalized last method for KERNEL perceptron:  %s:\n%s\n", classification_report(y_test, z))

#VOTE METHOD
#for n in range(len(X_test)):
    #for i in range(0,9):
        #score[i] = 0
        #for j in range(k[i]):
            #score[i]=score[i] + cost[i][j] * np.sign(np.dot(weights[i][j], X_test[n]))
    #z[n] = np.argmax(score)
    #print(z[n])

#print(z)

#print('Accuracy score for vote method ', accuracy_score(y_test, z))

#KERNEL VOTE METHOD
#for n in range(len(X_test)):
    #for i in range(0,10):
        #score[i] = 0
        #for j in range(k2[i]):
            #if j == 0:
                #score[i] = 0
            #else:
                #sign = np.sign(np.dot(weights2[i][j - 1], X_test[n]) + WY2[i][j - 1] * kvp.kernel(WX2[i][j - 1], X_test[n], d= dimension))
                #score[i] = score[i] + cost2[i][j] * sign
    #z2[n] = np.argmax(score)
    #print("valore atteso :", y_test[n], "valore indovinato:", z2[n])
    #print(z[n])

#print(z2)

#print('Accuracy score for vote method for KERNEL perceptron', accuracy_score(y_test, z2))
#print("Classification report for vote method for KERNEL perceptron:  %s:\n%s\n", classification_report(y_test, z2))

#AVERAGE (UNNORMALIZED) METHOD
#for n in range(len(X_test)):
    #for i in range(0,9):
        #score[i] = 0
        #for j in range(k[i]):
            #score[i]=score[i] + cost[i][j] * np.dot(weights[i][j], X_test[n])
    #z[n] = np.argmax(score)

#print(z)

#print('Accuracy score for unnormalized method ', accuracy_score(y_test, z))

#KERNEL AVERAGE (NORMALIZED) METHOD
# for n in range(len(X_test)):
#     for i in range(0,10):
#         score[i] = 0
#         for j in range(k2[i]):
#             if j == 0:
#                 score[i] = 0
#             else:
#                 sign = (np.dot(weights2[i][j - 1], X_test[n]) + WY2[i][j - 1] * kvp.kernel(WX2[i][j - 1], X_test[n], d= dimension))/lng.norm(weights2[i][j], 2)
#                 score[i] = score[i] + cost2[i][j] * sign
#     z2[n] = np.argmax(score)
#     #print("valore atteso :", y_test[n], "valore indovinato:", z2[n])
#     #print(z2[n])
#
# print(z2)
#
# print('Accuracy score for normalized average method for KERNEL perceptron', accuracy_score(y_test, z2))
# print("Classification report for normalized average  method for KERNEL perceptron:  %s:\n%s\n", classification_report(y_test, z2))

#AVERAGE (NORMALIZED) METHOD
#for n in range(len(X_test)):
    #for i in range(0,9):
        #score[i] = 0
        #for j in range(k[i]):
            #score[i]=score[i] + cost[i][j] * (np.dot(weights[i][j], X_test[n])/lng.norm(weights[i][j], 2))
    #z[n] = np.argmax(score)

#print(z)

#print('Accuracy score for normalized method ', accuracy_score(y_test, z))