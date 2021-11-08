# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 21:14:01 2021

@author: ahmed
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = 'E:\\belongs to graduation project\\ml projects\\linear regression\\linear reg.txt'
data = pd.read_csv(path,header=None,names=['population','people'])
data.plot(kind='scatter',x= 'population',y='people',figsize=(3,3))

data.insert(0,'onnnnes',1)
#print(data.head(10))


cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
#converting x y lists to matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
#print(X)

def computeCost(X, y, theta):
    # the list allsum will be summed through the return of fun computeCost 
    allsum = np.power(((X * theta.T) - y), 2)
    return np.sum(allsum) / (2 * len(X))
#print(computeCost(X, y, theta))


# this function will return the costs list through x iteration and list of theta  
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100

# perform linear regression on the data set
thetaa, cost = gradientDescent(X, y, theta, alpha, iters)
print("thetaa============",thetaa)

# sizzer
x = np.linspace(data.population.min(), data.population.max(), 100)

#best fit line result (matrix)
f = thetaa[0, 0] + (thetaa[0, 1] * x)

# draw the line  for population people

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.population, data.people, label='actual data')
ax.legend(loc=2)
ax.set_xlabel('population')
ax.set_ylabel('people')
ax.set_title('people vs. population')



# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training iter hypo')
