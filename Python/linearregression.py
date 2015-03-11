# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 17:02:53 2014

@author: lagosaurusrex
"""
import numpy as np
#np.seterr(divide="ignore",invalid="ignore")
import matplotlib
import pylab
from scipy import linalg

def xy_and_theta(sample_file):
    '''Takes a file and converts its contents into x and y numpy arrays.
       It also creates theta initialized to zeros as a numpy array.'''
    data = open(sample_file)
    x_data = []
    y_data = []
    for example in data.readlines():
        example = example.split(',')
        length = len(example)
        temp = [1]
        for dummy_x in xrange(length-1):
            temp.append(example[dummy_x])
        x_data.append(temp)
        y_data.append([example[length-1]])
    x_data = np.array(np.array(x_data, dtype=np.float64))
    y_data = np.array(np.array(y_data, dtype=np.float64))
    theta = np.zeros(length)
    return x_data, y_data, theta

def regularization(x):
    '''Regularizes the values of the features in X. First subtracting the value
       from the mean of values in that column, then dividing the new value by the 
       standard deviation of values in that column.'''
    length = len(x)
    l2 = len(x.transpose())
    new_x = []
    ct = 0
    for i in range(length):
        tmp = []
        for value in x[i]:
            if value == 1:
                tmp.append(1)
                ct+=1
            else:
                z = (value-np.mean(x[:,ct%l2]))/np.std(x[:,ct%l2])
                tmp.append(z)
                ct+=1
        new_x.append(tmp)
        tmp = []
    return np.array(new_x,np.float)

def compute_cost(x_data,y_data,theta):
    '''Computes the cost function J(0).
	1/2m * sum(h0(x^(i)-y^(i))^2''' 
    m = len(y_data)    
    hypothesis = x_data*theta
    sqrErrors = (hypothesis - y_data)**2
    return (1.000/(2.00*m)*(sum(sqrErrors)))[0]
    
def gradient_descent(x,y,theta,alpha,num_iter):
    '''Takes x, y and theta as matrices, along with the learning rate (alpha)
       and the number of iterations you'd like to run and minimizes the cost 
       function J(Theta) by using gradient descent.
	d/d*Thetaj * 1/2m * sum(hTheta(x^[i])-y^[i])**2  :
	Theta0 = 1/m * sum(hTheta(x^[i]) - y^[i])
	Theta1 = 1/m * sum(hTheta(x^[i]) - y^[i])*x^[i]
	'''
    m = len(y)
    j_history = []
    for dummy_x in xrange(num_iter):
        difference = x.dot(theta.transpose()) - y
        summe = difference.transpose().dot(x)
        theta = theta - summe * (alpha/m)
    return theta[0]
    
def normal_equation(x,y):
    '''The Normal Equation returns the optimum cost function, J(0).'''
    return linalg.inv((x.transpose().dot(x))).dot(x.transpose()).dot(y).flatten()

def predict(x,theta,x_data=None):
    '''Predict whether the label is 0 or 1 using LEARNED linear
       regression parameters.'''
    if x_data == None:
        return np.array(x).dot(np.array(theta))
    else:
        #Make sure to normalize the values of x if you normalized your features.
        for i in range(1,len(x)):
            x[i] = (x[i]-np.mean(x_data[:,i]))/np.std(x_data[:,i])
        return np.array(x).dot(np.array(theta))

if __name__ == '__main__':
    def test1(data):
        print "Data Set 1: Truck Profit per City Population"
        print "********************************************"
        x_data,y_data,theta = xy_and_theta(data)
        print "Compute Initial Cost: ", compute_cost(x_data,y_data,theta)
        print "Gradient Descent: ", gradient_descent(x_data,y_data,theta,.01,1500)
        print "Normal Equation: ", normal_equation(x_data,y_data)
        print "Predictions of the profit of a truck in a city with a population of 70,000."
        theta = gradient_descent(x_data,y_data,theta,.02,1500)
        print "Gradient Descent Prediction: $", predict([1,7],theta)*10000
        theta = normal_equation(x_data,y_data)
        print "Normal Equation Prediction: $", predict([1,7],theta)*10000
        print
    def test2(data):
        print "Data Set 2: Housing Prices per house size and number of bedrooms."
        print "*****************************************************************"
        x_data,y_data,theta = xy_and_theta(data)
        x_data2 = regularization(x_data)
        print "Compute Initial Cost: ", compute_cost(x_data2,y_data,theta)
        print "Gradient Descent: ", gradient_descent(x_data2,y_data,theta,.01,400)
        print "Normal Equation: ", normal_equation(x_data,y_data)
        print "Predictions of the profit of a truck in a city with a population of 70,000."
        theta = gradient_descent(x_data2,y_data,theta,.01,400)
        print "Gradient Descent Prediction: $", predict([1,1650,3],theta,x_data)
        theta = normal_equation(x_data,y_data)
        print "Normal Equation Prediction: $", predict([1,1650,3],theta)
        print
    test1("ex1data1.txt")
    test2("ex1data2.txt")