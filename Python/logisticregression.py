# -*- coding: utf-8 -*-
"""
Created on Sun Aug 03 02:27:09 2014

@author: LAGosaurusRex
"""
import numpy as np
#from scipy.optimize import fmin_bfgs
import scipy.optimize as op
from linearregression import normal_equation

def sigmoid(z):
    return 1.0/(1.0+np.e**(-z))

def cost(theta,x,y):
    m,n = x.shape
    theta = theta.reshape((n,1))
    _hypothesis = sigmoid(np.dot(x,theta))
    _log_error = y*np.log(_hypothesis)+(1-y)*np.log(1-_hypothesis)
    cost = -1.0/m*sum(_log_error)
    return cost

def gradient(theta,x,y,lamb=1.0):
    m,n = x.shape
    theta = theta.reshape((n,1))
    _hypothesis = sigmoid(np.dot(x,theta))
    #Compute Gradient
    grad = np.zeros(theta.shape)
    for i in range(m):
        new_x = x[i,:]
        new_x.shape = (len(new_x),1)
        grad = grad + (_hypothesis[i]-y[i])*(new_x)
    return ((1.0/m)*grad).transpose().flatten()
    
def predict(theta,x):
    '''Predict whether the label is 0 or 1 using LEARNED logistic regression
       parameters theta.'''
    m, n = x.shape
    p = np.zeros(shape=(m, 1))

    h = sigmoid(x.dot(theta.T))

    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0
    return p
    
if __name__ == '__main__':
    def test1(data):
        data = np.loadtxt(data,delimiter=',')
        x = data[:,0:2]
        y = data[:,2]
        y.shape = (len(y),1)
        x = np.insert(x,0,1,1)
        [m,n] = x.shape
        #Initialize Theta
        initial_theta = np.zeros([n,1])
        #Compute and display initial cost and gradient
        print "Cost at initial theta (zeros): ", cost(initial_theta,x,y)
        print "Gradient at initial theta (zeros): ", gradient(initial_theta,x,y)
        theta = op.minimize(cost,initial_theta,(x,y),'TNC',jac=gradient).x
        print "Theta found through scipy.optimize.minimize: ", theta
        prob = sigmoid(np.array([1.0, 45.0, 85.0]).dot(np.array(theta).T))
        print 'For a student with scores 45 and 85, we predict an admission ' + \
            'probability of %f' % prob
        #Compute accuracy on our training set
        p = predict(np.array(theta), x)
        print 'Train Accuracy: %f' % ((y[np.where(p == y)].size / float(y.size)) * 100.0)
    test1('ex2data1.txt')
