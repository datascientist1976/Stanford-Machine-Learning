# -*- coding: utf-8 -*-
"""
Created on Fri Aug 08 11:16:52 2014

@author: LAGosaurusRex
"""
import numpy as np
import scipy.io as sio
import scipy.optimize as op

def sigmoid(z):
    return 1.0/(1.0+np.e**(-z))

def randinitializeweights(L_in,L_out):
    '''Randomly initialize the weights of a layer with L_in incoming connections
       and L_out outgoing connections.'''
    epsilon = 0.12
    w = np.random.rand(L_out,1+L_in)*2*epsilon-epsilon
    return w

def sigmoidgradient(z):
    '''Returns the gradient of the sigmoid function evaluated at z.
       Works for a Matrix, Vector, or a Scalar Value'''
    #Scalar
    if type(z) == int:
        return sigmoid(z) * (1-sigmoid(z))
    z = np.array(z)
    if len(np.shape(z)) < 2:
        z = z.reshape(len(z),1)
    m,n = np.shape(z)
    g = np.zeros(np.shape(z))
    #Matrix
    if n > 2:
        for i in range(m):
            for j in range(n):
                g[i,j] = sigmoid(z[i,j])*(1-sigmoid(z[i,j]))
    #Vector
    else:
        for i in range(len(z)):
            g[i] = sigmoid(z[i]) * (1-sigmoid(z[i]))
    return g
    
def cost(nnparams,hidden_layer_size,num_labels,x,y,lamb=.1):
    '''Compute the Cost function for the Neural Network'''
    m,n = np.shape(x)
    input_layer_size = n
    # Reshape nnparams back into Theta1 and Theta2
    # The weight matrices for our 2 layer neural network
    theta1 = np.array(nnparams[0:hidden_layer_size*(input_layer_size+1)]).reshape(hidden_layer_size,(input_layer_size+1))
    theta2 = np.array(nnparams[hidden_layer_size*(input_layer_size+1):]).reshape(num_labels,hidden_layer_size+1)
    J=0
    for i in xrange(m):
        index = y[i][0]-1
        #Forward Propagation
        a1 = x[i,:].transpose()
        a1 = np.insert(a1,0,1,0)
        z2 = np.dot(theta1,a1)
        a2 = sigmoid(z2)
        a2 = np.insert(a2,0,1,0)
        z3 = np.dot(theta2,a2)
        a3 = sigmoid(z3)
        a3 = a3.reshape(len(a3),1)
        #Initialize y
        yk = np.zeros([num_labels,1])
        yk[index] = 1
        #Compute Cost
        ht = a3
        s1 = -1*yk*np.log(ht)
        s2 = -1*(1-yk)*np.log(1-ht)
        sk = sum(s1+s2)
        J = J+sk
    J=J/m
    #Regularize the Cost
    t1 = theta1
    t1 = t1[1:]
    t1 = np.insert(t1,0,0,0)
    t1_square = t1*t1
    st1 = sum(sum(t1_square))
    t2 = theta2
    t2 = t2[1:]
    t2 = np.insert(t2,0,0,0)
    t2_square = t2*t2
    st2 = sum(sum(t2_square))
    J = J+lamb/(2*m)*(st1+st2)    
    return J[0]
    
def gradient(nnparams,hidden_layer_size,num_labels,x,y,lamb=.1):
    '''Computes the Gradient of the Neural Network.'''
    m,n = np.shape(x)
    input_layer_size = n
    # Reshape nnparams back into Theta1 and Theta2
    # The weight matrices for our 2 layer neural network
    theta1 = np.array(nnparams[0:hidden_layer_size*(input_layer_size+1)]).reshape(hidden_layer_size,(input_layer_size+1))
    theta2 = np.array(nnparams[hidden_layer_size*(input_layer_size+1):]).reshape(num_labels,hidden_layer_size+1)
    theta1_grad = np.zeros(np.shape(theta1))
    theta2_grad = np.zeros(np.shape(theta2))
    for i in xrange(m):
        index = y[i][0]-1
        #Forward Propagation
        a1 = x[i,:].transpose()
        a1 = np.insert(a1,0,1,0)
        z2 = np.dot(theta1,a1)
        a2 = sigmoid(z2)
        a2 = np.insert(a2,0,1,0)
        z3 = np.dot(theta2,a2)
        a3 = sigmoid(z3)
        a3 = a3.reshape(len(a3),1)
        # Initialize y
        yk = np.zeros([num_labels,1])
        yk[index] = 1
        #Backpropagation
        delta_3 = (a3-yk)
        delta_2 = np.dot(theta2.transpose(),delta_3)
        delta_2 = delta_2[1:]
        delta_2 = delta_2*sigmoidgradient(z2)
        #Gradients
        theta2_grad = theta2_grad + (delta_3*a2.transpose())
        theta1_grad = theta1_grad + (delta_2*a1.transpose())
    t1 = theta1
    t1 = t1[1:]
    t1 = np.insert(t1,0,0,0)
    t2 = theta2
    t2 = t2[1:]
    t2 = np.insert(t2,0,0,0)
    # Compute the Regularized Gradient
    theta1_grad = theta1_grad/m+lamb/m*t1
    theta2_grad = theta2_grad/m+lamb/m*t2
    # Unroll the Thetas into a vector
    theta1 = theta1_grad.reshape(1,(input_layer_size+1)*hidden_layer_size)
    theta2 = theta2_grad.reshape(1,(hidden_layer_size+1)*num_labels)
    #print "RIGHT HERE!!!", theta2
    nnparams = []
    for i in theta1[0]:
        nnparams.append(i)
    for i in theta2[0]:
        nnparams.append(i)
    return np.array(nnparams)
    
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
    if len(a3) < 2:
        return a3
    return a3.index(max(a3))
    
def unroll_params(theta1,theta2,hidden_layer_size,input_layer_size,num_labels):
    theta1 = theta1.reshape(1,(input_layer_size+1)*hidden_layer_size)
    theta2 = theta2.reshape(1,(hidden_layer_size+1)*num_labels)
    nnparams = []
    for i in theta1[0]:
        nnparams.append(i)
    for i in theta2[0]:
        nnparams.append(i)
    nnparams = np.array(nnparams)
    return nnparams
    
def get_thetas(nnparams,hidden_layer_size,input_layer_size,num_labels):
    theta1 = np.array(nnparams[0:hidden_layer_size*(input_layer_size+1)]).reshape(hidden_layer_size,(input_layer_size+1))
    theta2 = np.array(nnparams[hidden_layer_size*(input_layer_size+1):]).reshape(num_labels,hidden_layer_size+1)   
    return theta1,theta2

def test():
    mat = sio.loadmat('ex3data1.mat')
    x = mat['X']
    y = mat['y']
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    theta1 = randinitializeweights(input_layer_size,hidden_layer_size)
    theta2 = randinitializeweights(hidden_layer_size,num_labels)
    theta1 = theta1.reshape(1,(input_layer_size+1)*hidden_layer_size)
    theta2 = theta2.reshape(1,(hidden_layer_size+1)*num_labels)
    nnparams = []
    for i in theta1[0]:
        nnparams.append(i)
    for i in theta2[0]:
        nnparams.append(i)
    nnparams = np.array(nnparams)
    print "new cost function: ",cost(nnparams,hidden_layer_size,num_labels,x,y)
    #print "new cost gradient: ",gradient(nnparams,hidden_layer_size,num_labels,x,y)
    #nnparams = gradient(nnparams,hidden_layer_size,num_labels,x,y)
    #nnparams = op.fmin_bfgs(cost, nnparams, maxiter=100, args=(hidden_layer_size,num_labels,x,y,0),fprime=gradient)
    nnparams = op.fmin_cg(cost, nnparams, maxiter=100, args=(hidden_layer_size,num_labels,x,y,.001),fprime=gradient)
    #nnparams = op.minimize(cost,nnparams,(hidden_layer_size,num_labels,x,y,0.1),'TNC',jac=gradient).x
    theta1 = np.array(nnparams[0:hidden_layer_size*(input_layer_size+1)]).reshape(hidden_layer_size,(input_layer_size+1))
    theta2 = np.array(nnparams[hidden_layer_size*(input_layer_size+1):]).reshape(num_labels,hidden_layer_size+1)
    print "Running Predictions."
    pred1 = predict(theta1,theta2,np.array(x[0]).reshape(1,400))
    print "This value should be 10: ",pred1+1
    correct = 0
    counter = 0
    for i in range(len(y)):
        #print "Digit is a: ", y[i][0]
        pred = predict(theta1,theta2,np.array(x[i]).reshape(1,400))
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

if __name__ == '__main__':
    test()
        
    
