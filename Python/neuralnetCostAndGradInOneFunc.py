# -*- coding: utf-8 -*-
"""
Created on Sat Aug 09 02:19:26 2014

@author: LAGosaurusRex
"""
import numpy as np
import reglogisticregression as rl2
import oneVsAll as OVA
import scipy.io as sio
import scipy.optimize as op

def neuralnetcost(nnparams,hidden_layer_size,num_labels,x,y,lamb=1):
    m,n = np.shape(x)
    input_layer_size = n
    # Initialize the Thetas
    theta1 = np.array(nnparams[0:hidden_layer_size*(input_layer_size+1)]).reshape(hidden_layer_size,(input_layer_size+1))
    theta2 = np.array(nnparams[hidden_layer_size*(input_layer_size+1):]).reshape(num_labels,hidden_layer_size+1)
    theta1_grad = np.zeros(np.shape(theta1))
    theta2_grad = np.zeros(np.shape(theta2))
    J=0
    for i in xrange(m):
        index = y[i][0]-1
        #Forward Propagation
        a1 = x[i,:].transpose()
        a1 = np.insert(a1,0,1,0)
        z2 = np.dot(theta1,a1)
        a2 = rl2.sigmoid(z2)
        a2 = np.insert(a2,0,1,0)
        z3 = np.dot(theta2,a2)
        a3 = rl2.sigmoid(z3)
        a3 = a3.reshape(len(a3),1)
        
        yk = np.zeros([num_labels,1])
        yk[index] = 1
        #Compute Delta
        delta_3 = (a3-yk)
        delta_2 = np.dot(theta2.transpose(),delta_3)
        delta_2 = delta_2[1:]
        delta_2 = delta_2*sigmoidgradient(z2)
        #Gradients
        #print np.shape(delta_3),np.shape(a2.reshape(1,len(a2)).transpose())
        theta2_grad = theta2_grad + delta_3*a2.transpose()
        theta1_grad = theta1_grad + delta_2*a1.transpose()
        #Sum of the cost
        ht = a3
        s1 = -1*yk*np.log(ht)
        s2 = -1*(1-yk)*np.log(1-ht)
        sk = sum(s1+s2)
        J = J+sk
    J=J/m
    print J[0]
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
    theta1_grad = theta1_grad/m+lamb/m*t1
    theta2_grad = theta2_grad/m+lamb/m*t2
    thetas = [theta1_grad,theta2_grad]
    
    theta1 = theta1.reshape(1,(input_layer_size+1)*hidden_layer_size)
    theta2 = theta2.reshape(1,(hidden_layer_size+1)*num_labels)
    nnparams = []
    for i in theta1[0]:
        nnparams.append(i)
    for i in theta2[0]:
        nnparams.append(i)
    
    return J[0],np.array(nnparams)
    
def debuginitialize(fan_out,fan_in):
    w = np.zeros([fan_out,1+fan_in])
    w = np.array(np.reshape(np.sin(range(np.size(w))),np.size(w)))/10
    return w
    
def test_net(lamb=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 5
    m = 5
    # Set *random* test data
    Theta1 = debuginitialize(hidden_layer_size,input_layer_size).reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = debuginitialize(num_labels,hidden_layer_size).reshape(num_labels,hidden_layer_size+1)
    x = debuginitialize(m,input_layer_size-1).reshape(m,input_layer_size)
    y = np.array(1 + np.mod(range(m),num_labels).transpose())
    y = y.reshape(len(y),1)
    # Unroll Params
    Theta1 = Theta1.reshape(1,(input_layer_size+1)*hidden_layer_size)
    Theta2 = Theta2.reshape(1,(hidden_layer_size+1)*num_labels)
    nnparams = []
    for i in Theta1[0]:
        nnparams.append(i)
    for i in Theta2[0]:
        nnparams.append(i)
    grad = op.fmin_cg(cost, nnparams, maxiter=100, args=(hidden_layer_size,num_labels,x,y,.001),fprime=gradient)
    numgrad = computeNumGrad(nnparams,hidden_layer_size,num_labels,x,y,lamb=.1)
    
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('''If your backprop implementation is correct, then 
    'the relative difference will be small (less than 1e-9. \'
    '\n Relative Difference: '''), diff
    
def computeNumGrad(theta,hidden_layer_size,num_labels,x,y,lamb=.1):
    numgrad = np.zeros(np.shape(theta))
    perturb = np.zeros(np.shape(theta))
    e = 1e-4
    for p in range(np.size(theta)):
        perturb[p] = e
        loss1 = op.minimize(cost,theta-perturb,(hidden_layer_size,num_labels,x,y,.1),'TNC',jac=gradient).x
        loss2 = op.minimize(cost,theta+perturb,(hidden_layer_size,num_labels,x,y,.1),'TNC',jac=gradient).x
        numgrad[p] = (sum(loss2)-sum(loss1))/(2*e)
        perturb[p] = 0
    return numgrad