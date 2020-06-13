import numpy as np
import mnist_reader
import KernelVotedPerceptron as kvp
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


#we load the complete dataset
X1, y1 = mnist_reader.load_mnist('data/fashion', kind='train')
X2, y2 = mnist_reader.load_mnist('data/fashion', kind='t10k')


X_train = []
y_train = []

X_test = []
y_test = []

class1 = 1
class2 = 8

#we load the elements of the 2 classes in our train and test set
for i in range(len(y1)):
    if y1[i] == class1 or y1[i] == class2:
        X_train.append(X1[i])
        if y1[i] == class1:
            y_train.append(-1)
        else:
            y_train.append(1)

for i in range(len(y2)):
    if y2[i] == class1 or y2[i] == class2:
        X_test.append(X2[i])
        if y2[i] == class1:
            y_test.append(-1)
        else:
            y_test.append(1)


#normalize the data
X_train[:] = [ x / 255 for x in X_train]
X_test[:] = [ x / 255 for x in X_test]

#this vector will be used to plot both the test error and the learning curve
plot_epochs = [i for i in np.arange(0.1,10,0.1)]
#this is for the learning curve
errors_during_training = []

for dimension in range(1,4):     #we start from dimension = 1 up to dimension = 3


    errors = []

    kernel_voted_perceptron = kvp.KernelVotedPerceptron(T=10, d = dimension)

    print('Training the voted perceptron in dimension = ', dimension, 'on classes ', class1, 'and ', class2)

    partial = kernel_voted_perceptron.train(X_train, y_train)

    errors_during_training.append(partial)        #we save the number of errors in that dimension for every tenth of an epoch

    print('Training completed')

    predictions = kernel_voted_perceptron.vote(X_test, partial)

    #we calculate the number of mistaken labels on the test set for every tenth of an epoch and we normalize it to the total number of test instances
    for i in range(len(predictions[0])):
        error = 0
        for j in range(len(predictions)):
            if predictions[j][i] != y_test[j]: error += 1
        errors.append(error * 100/ len(predictions))

    #this vector represent the prediction using the final version of the weight vector
    final_prediction = [row[-1] for row in predictions]

    print('Accuracy score of vote method on test labels in dimension', dimension, ':', accuracy_score(y_test, final_prediction) * 100, '%')
    print('Classification report for prediction in dimension', dimension, ': \n\n', classification_report(y_test, final_prediction))
    print()

    #plot the test error for this dimension
    plt.rcParams['font.size'] = 8
    plt.title('Test Error in d = ' + str(dimension) + ' on classes ' + str(class1) + ' and ' + str(class2))
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    plt.plot(plot_epochs, errors, label = 'dimension = 1')
    plt.show()

#plot the learning curve for the 3 dimensions
plt.rcParams['font.size'] = 8
plt.title('Learning rate in training on classes ' + str(class1) + ' and ' + str(class2))
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.plot(plot_epochs, errors_during_training[0], label = 'dimension = 1')
plt.plot(plot_epochs, errors_during_training[1], label = 'dimension = 2')
plt.plot(plot_epochs, errors_during_training[2], label = 'dimension = 3')
plt.legend()
plt.show()
