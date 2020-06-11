import mnist_reader
import KernelVotedPerceptron as kvp
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

#function to permute the training set
def permute(preX, preY):
    supportmatrix = []
    for i in range(len(preY)):
        support = []
        for j in range(len(preX[i])):
            support.append(preX[i][j])
        support.append(preY[i])
        supportmatrix.append(support)

    np.random.shuffle(supportmatrix)

    Y_train = []
    X_train = []

    for i in range(len(supportmatrix)):
        Y_train.append(np.array(supportmatrix[i][-1]))
        X_train.append(np.array(supportmatrix[i][0:-1]))

    return X_train, Y_train

#we load the complete dataset
X1, Y1 = mnist_reader.load_mnist('data/fashion', kind='train')
X2, Y2 = mnist_reader.load_mnist('data/fashion', kind='t10k')

#this for cycle is for the 2 example of binary classifications
for run in range(2):

    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    preX = []
    preY = []

    #each time we select 2 different classes for the perceptron
    if run == 0:
        class1 = 0
        class2 = 5
    if run == 1:
        class1 = 1
        class2 = 9

    #we select the elements of the 2 chosen classes from the original dataset
    for i in range(len(Y1)):
        if Y1[i] == class1 or Y1[i] == class2:
            preX.append(X1[i])
            if Y1[i] == class1:
                preY.append(-1)
            else:
                preY.append(1)

    for i in range(len(Y2)):
        if Y2[i] == class1 or Y2[i] == class2:
            X_test.append(X2[i])
            if Y2[i] == class1:
                Y_test.append(-1)
            else:
                Y_test.append(1)

    #we normalize the values
    preX[:] = [x / 255 for x in preX]
    X_test[:] = [ x / 255 for x in X_test]

    X_train, Y_train = permute(preX, preY)

    #creating 2 vector to plot for the Test Errors Graph
    plot_epochs = []
    for i in range(200):
        plot_epochs.append(i * 1 / 25)

    errors = []

    for dimension in range(1,4):     #we start from dimension = 1 up to dimension = 3

        kernel_voted_perceptron = kvp.KernelVotedPerceptron(T=10, d = dimension)

        print('Training the voted perceptron on classes: ', class1, 'and: ', class2, 'in dimension = ', dimension)

        kernel_voted_perceptron.train(X_train, Y_train)

        print('Training completed')
        errors.append(kernel_voted_perceptron.errors)

        predictions = kernel_voted_perceptron.vote(X_test)

        print('Accuracy score of the prediction on test labels in dimension', dimension, ':', accuracy_score(Y_test, predictions))
        print('Classification report for prediction in dimension', dimension, ': \n\n', classification_report(Y_test, predictions))
        print()

    #plotting the errors-per-epoch function
    plt.rcParams['font.size'] = 8
    plt.title('Errors during training on classes ' + str(class1) + ' and ' + str(class2))
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.plot(plot_epochs, errors[0], label = 'dimension = 1')
    plt.plot(plot_epochs, errors[1], label = 'dimension = 2')
    plt.plot(plot_epochs, errors[2], label = 'dimension = 3')
    plt.legend()
    plt.show()
