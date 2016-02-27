# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import processData as pd
import normalize as norm
from bigfloat import *
import bigfloat
from matplotlib.legend_handler import HandlerLine2D

# Helper function to evaluate the total loss on the dataset
def calculate_loss(X, y, model):
    num_examples = X.shape[0]
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    #exp_scores = bigfloat.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    #exp_scores = bigfloat.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, costList, nn_hdim, num_passes= 2800, print_loss=False):

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        #exp_scores = bigfloat.exp(z2)
        #print(exp_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        #delta3 = delta3 - y
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 30 == 0:
            cost = calculate_loss(X, y, model)
            costList.append(cost)
            print("Loss after iteration %i: %f" %(i, cost))

    return model


# Display plots inline and change default figure size
'''
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
print(X.shape)
print(y)
'''

X = pd.processData('adult.csv')

y = np.matrix(np.zeros((X.shape[0],2),dtype='int'))
y = np.array(X[:, -1],dtype='int')

X = X[:, 0:X.shape[1] - 1]
X = norm.normalizeStd(X)

#Define training set and cross validation set
n = X.shape[0]
X_training = X[0: 0.9 * n, :]
y_training = y[0: 0.9 * n]

y_cross = y[0.9 * n : n]
X_cross = X[0.9 * n:n, :]


num_examples = X_training.shape[0] # training set size
nn_input_dim = X_training.shape[1] # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

# Gradient descent parameters (I picked these by hand)
epsilon = 0.0001 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength
num_iter = 200
costONTrainingSet = []

# Build a model with a 3-dimensional hidden layer\
model = build_model(X_training, y_training, costONTrainingSet, 5,num_iter, print_loss=True)

predictions = predict(model, X_training)

loss_train = calculate_loss(X_training, y_training, model)
title = 'Training set loss = ' + str("%.5f" % loss_train)
print(title)

predictions = predict(model, X_cross)
plt.figure(1)
xaxis = np.arange(0, len(y_cross))
plt.plot(xaxis, predictions, 'x' , label='cross set predictions')
plt.plot(xaxis, y_cross, 'x' , label ='real result')
#loss_cross = np.square(predictions - y_cross).sum() / len(y_cross)
loss_cross = calculate_loss(X_cross, y_cross, model)
title = 'Cross Validation loss = ' + str("%.5f" % loss_cross)
print(title)
plt.title(title)
y_crosslagend = plt.xlabel("Cross validation Real result")
plt.ylabel("index of examples")
#plt.legend(handler_map={y_crosslagend: HandlerLine2D(numpoints=4)})

plt.figure(2)
xaxis = np.arange(0, num_iter / 30)
plt.plot(xaxis, costONTrainingSet)
title = 'Cost of Training set, cost =  ' + str(costONTrainingSet[-1])
plt.title(title)
plt.ylabel("loss")
plt.xlabel("Number of iteration")

#run on test set
data_test = pd.processData('adult.test')

y_test = np.array(data_test[:, -1],dtype='int')
X_test = data_test[:, 0:data_test.shape[1] - 1]

X_test = norm.normalizeStd(X_test)

predictions = predict(model, X_test)

loss_test = calculate_loss(X_test, y_test, model)
title = 'Test set MSE = ' + str("%.5f" % loss_test)
print(title)


#try different hiden size
loss_diff_hidensize = []
loss_diff_hidensize_cross = []
for hidensize in range(3, 40):
    model = build_model(X_training, y_training, costONTrainingSet, hidensize ,num_iter, print_loss=False)

    predictions = predict(model, X_training)
    loss = calculate_loss(X_training, y_training, model)
    title = 'Training set loss  = ' + str("%.5f" % loss) + ' with hiden size = ' + str(hidensize)
    print(title)
    loss_diff_hidensize.append(loss)

    predictions = predict(model, X_cross)
    loss = calculate_loss(X_cross, y_cross, model)
    title = 'Cross validation set loss = ' + str("%.5f" % loss) + ' with hiden size = ' + str(hidensize)
    print(title)
    loss_diff_hidensize_cross.append(loss)

plt.figure(4)
xaxis = np.arange(3, 40)
plt.plot(xaxis, loss_diff_hidensize)
plt.plot(xaxis, loss_diff_hidensize_cross)
title = 'cross validation error and training set error '
plt.title(title)
plt.ylabel("loss")
plt.xlabel("hiden layer size")


###########################################################################
loss_diffsize_train = []
loss_diffsize_cross = []
costONTrainingSet = []
for j in range(1, np.int(X_training.shape[0] /1000)):
    i = j * 1000
    X_diffsize = X_training[0:i, :]
    y_diffsize = y_training[0:i]
    num_examples = X_diffsize.shape[0] # training set size
    nn_input_dim = X_diffsize.shape[1] # input layer dimensionality
    X_diffsize_train = X_diffsize[0:0.9*num_examples, :]
    y_diffsize_train = y_diffsize[0:0.9*num_examples]

    X_diffsize_cross = X_diffsize[0.9*num_examples:num_examples, :]
    y_diffsize_cross = y_diffsize[0.9*num_examples : num_examples]

    num_examples = X_diffsize_train.shape[0] # training set size
    nn_input_dim = X_diffsize_train.shape[1] # input layer dimensionality
    curmodel = build_model(X_diffsize_train, y_diffsize_train, costONTrainingSet, 3,num_iter, print_loss=False)

    predictions = predict(curmodel, X_diffsize_train)

    loss_diffsize_train.append(calculate_loss(X_diffsize_train, y_diffsize_train, model))

    predictions = predict(curmodel, X_diffsize_cross)

    loss_diffsize_cross.append(calculate_loss(X_diffsize_cross, y_diffsize_cross, model))
#####################################################################################################

plt.figure(3)
xaxis = np.arange(1, np.int(X_training.shape[0] /1000))
plt.plot(xaxis, loss_diffsize_cross)
plt.plot(xaxis, loss_diffsize_train)
title = 'cross validation error and training set error '
plt.title(title)
plt.ylabel("loss")
plt.xlabel("size of the tainning data")

# Train the logistic rgeression classifier
clf = sklearn.linear_model.LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
           fit_intercept=True, intercept_scaling=1.0, max_iter=100,
           multi_class='ovr', n_jobs=1, penalty='l2', refit=True,
           scoring=None, solver='lbfgs', tol=0.0001, verbose=0)
clf.fit(X_training, y_training)



predictions = clf.predict(X_cross)
loss_cross_lr = calculate_loss(X_cross, y_cross, model)
title = 'Logistic RegressionCV cross validation loss = ' + str("%.5f" % loss_cross_lr)
print(title)

predictions = clf.predict(X_test)
loss_test_lr = calculate_loss(X_test, y_test, model)
title = 'Logistic RegressionCV test set loss = ' + str("%.5f" % loss_test_lr)
print(title)


plt.show()