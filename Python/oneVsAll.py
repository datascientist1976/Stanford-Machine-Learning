# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:31:40 2014

@author: lagosaurusrex
"""
import numpy as np
import scipy.io as sio
import scipy.optimize as op

def sigmoid(z):
    return 1.0/(1.0+np.e**(-z))

def cost(theta,x,y,lamb):
    m,n = x.shape
    theta = theta.reshape((n,1))
    _hypothesis = sigmoid(np.dot(x,theta))
    _log_error = y*np.log(_hypothesis)+(1-y)*np.log(1-_hypothesis)
    cost = -1.0/m*sum(_log_error) + lamb/(2*m)*sum(theta[1:])**2
    return cost

def gradient(theta,x,y,lamb):
    m,n = x.shape
    theta = theta.reshape((n,1))
    _hypothesis = sigmoid(np.dot(x,theta))
    #Compute Gradient
    return ((np.dot(x.transpose(),(_hypothesis-y)) + (lamb*theta)))/m

def oneVsAll(x,y,num_labels,lamb=0.0):
    '''Trains multiple logistic regression classifiers and returns all the
       classifiers in a matrix [all_theta], where the i-th row of all_theta 
       corresponds to the classifier for label i.'''
    n = np.size(x,1)
    # all_theta is what oneVsAll will be returning
    all_theta = np.zeros([num_labels,n+1])
    # Add a vector of ones at the beginning of X
    x = np.insert(x,0,1,1)
    for c in xrange(0,num_labels):
        if c == 0:
            index = 10
        else:
            index = c
        initial_theta = np.zeros(n+1)
        new_x = x
        new_y = np.zeros([len(new_x),1])
        for i in xrange(len(y)):
            if y[i][0] == index:
                new_y[i] = 1
        xtheta = op.minimize(cost,initial_theta,(new_x,new_y,0.1),'TNC',jac=gradient).x
        all_theta[c,:] = xtheta
    return all_theta
    
def predictOneVsAll(all_theta,x,y):
    '''Returns a list of tuples, with each tuple containing the trained prediction of a
       training example, and the output of the training example in y.'''
    x = np.insert(x,0,1,1)                           # Add the intercept term to x
    m = len(y)                                       # No. of training example
    # Vector of class classifications of length (K Classes).
    # The max of this vector is the class that best fits the training example.
    # i.e if the image is of a 1, it will attempt to output a vector like:
    # [0,1,0,0,0,0,0,0,0,0]
    c = sigmoid(np.dot(x,all_theta.transpose()))     
    predictions = []
    for i in xrange(m):
        temp = list(c[i])
        index = temp.index(max(temp))
        if index == 0:
            index = 10
        predictions.append((index,list(y[i])[0]))
    return predictions
    
def predict(Theta1,Theta2,x_sample):
    '''Neural Network Prediction on a single training example with two hidden layers ([Theta1],[Theta2]).
       Uses forward propagation to compute a vector of size K, (K = #Of Labels)'''
    x_sample = np.insert(x_sample,0,1,1) 
    a1 = x_sample
    #Compute the first hidden layer
    a2 = sigmoid(np.dot(a1,Theta1.transpose()))
    a2 = np.insert(a2,0,1,1)
    #Compute the second hidden layer
    a3 = sigmoid(np.dot(a2,Theta2.transpose()))
    a3 = list(a3.flatten())
    return a3.index(max(a3))

if __name__ == '__main__':
    mat = sio.loadmat('ex3data1.mat')
    x = mat['X']
    y = mat['y']
    num_labels = 10
    all_theta = oneVsAll(x,y,num_labels,0.1)
    trained_pred = predictOneVsAll(all_theta,x,y)
    correct = 0
    total = 0
    for pred in trained_pred:
        if pred[0] == pred[1]:
            correct += 1
            total += 1
        else:
            total += 1
    print "Training Set Accuracy: ",float(correct)/float(total)*100
    weights = sio.loadmat('ex3weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    correct = 0
    counter = 0
    for i in range(len(y)):
        #print "Digit is a: ", y[i][0]
        pred = predict(Theta1,Theta2,np.array(x[i]).reshape(1,400))
        if pred == 10:
            pred = 0
        else:
            pred = pred+1
        #print "Prediction on Digit is: ", pred
        if y[i][0] == pred:
            correct += 1
            counter += 1
        else:
            counter+=1
    print "Neural Network Accuracy: ",float(correct)/float(counter)*100
        