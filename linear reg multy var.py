# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 05:32:10 2021

@author: ahmed
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


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


#read data    
path =  'E:\\belongs to graduation project\\ml projects\\linear regression\\linear reg mult.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# rescaling data beacuase of big nums that may cause error , to reduce graph scale too
data = (data - data.mean()) / data.std()

# x0 that neded to mul ops
data.insert(0, 'Ones', 1)


# split data to be prepared to make operations
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))

alpha = 0.1
iters = 100

# perform linear regression on the data set the g and cost will needed to draw plots
g, cost = gradientDescent(X, y, theta, alpha, iters)

# get best fit line for Size vs. Price

x = np.linspace(data.Size.min(), data.Size.max(), 100)

f = g[0, 0] + (g[0, 1] * x)

# draw the line for Size vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Size, data.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')


# get best fit line for Bedrooms vs. Price

x = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

# draw the line  for Bedrooms vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Bedrooms, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')
###############################
# get best fit line for Bedrooms&Size vs. Price 
#NOTE THAT the lines between the ########## are under revising as searching about new way to plot 3d releationship

x1 = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)
x2 = np.linspace(data.Size.min(), data.Size.max(), 100)
f = g[0, 0] + (g[0, 1] * x1)+(g[0, 2] * x2) 

# draw the line  for Bedrooms&Size vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x1,x2, f, 'r', label='Prediction')
ax.scatter(data.Bedrooms,data.Size, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms with Size')
ax.set_ylabel('Price')
ax.set_title('Size&Bedrooms vs. Price')
###################################


# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

