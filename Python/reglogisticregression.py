# -*- coding: utf-8 -*-
"""
Created on Sun Aug 03 04:08:08 2014

@author: LAGosaurusRex
"""

import numpy as np
import scipy.optimize as op

def sigmoid(z):
    return 1.0/(1.0+np.e**(-z))

def cost(theta,x,y,lamb=1):
    m,n = x.shape
    theta = theta.reshape((n,1))
    _hypothesis = sigmoid(np.dot(x,theta))
    _log_error = y*np.log(_hypothesis)+(1-y)*np.log(1-_hypothesis)
    cost = -1.0/m*sum(_log_error) + lamb/(2*m)*sum(theta[1:])**2
    return cost

def gradient(theta,x,y,lamb=1):
    m,n = x.shape
    theta = theta.reshape((n,1))
    _hypothesis = sigmoid(np.dot(x,theta))
    #Compute Gradient
    grad = np.zeros(theta.shape)
    for i in range(m):
        new_x = x[i,:]
        new_x.shape = (len(new_x),1)
        grad = grad + (_hypothesis[i]-y[i])*(new_x)
    ta = np.insert(theta[1:],0,0,0)
    return ((1.0/m)*grad) + lamb/m*ta
    
def predict(theta, x):
    '''Predict whether the label is 0 or 1 using LEARNED logistic regression
       parameters theta.'''
    p_1 = sigmoid(np.dot(x, theta))
    return np.array(p_1 > 0.5).reshape(len(p_1),1)

def map_feature(features,degree):
    '''
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    Inputs X1, X2 must be the same size
    '''
    if len(features) % 2 == 0:
        temp = len(features)
    else:
        temp = len(features)-1
    new_features = []
    for x in range(0,temp,2):
        for i in range(1,degree+1):
            for j in range(i+1):
                new_features.append((float(features[x])**(float(i)-j))*(float(features[x+1])**float(j)))
    return new_features
        
def map_x(z,degree):
        new_x = []
        for i in z:
            new_x.append(map_feature(i,degree))
        new_x = np.array(new_x)
        return new_x
    
if __name__ == '__main__':
    def test1(data):
        data = np.loadtxt(data,delimiter=',')
        x = data[:,0:2]
        y = data[:,2]
        x = map_x(x,7)
        y.shape = (len(y),1)
        x = np.insert(x,0,1,1)
        #print x
        [m,n] = x.shape
        #Initialize Theta
        initial_theta = np.zeros([n,1])
        #Compute and display initial cost and gradient
        #print "Cost at initial theta (zeros): ", cost(initial_theta,x,y)
        #print "Gradient at initial theta (zeros): ", gradient(initial_theta,x,y)
        theta = op.minimize(cost,initial_theta,(x,y,1),'TNC',jac=gradient)
        #print theta
        theta = theta.x
        #Compute accuracy on our training set
        p = predict(np.array(theta), x)
        #p = predict2(np.array(theta), x)
        print 'Train Accuracy: %f' % ((y[np.where(p == y)].size / float(y.size)) * 100.0)
    test1('ex2data2.txt')