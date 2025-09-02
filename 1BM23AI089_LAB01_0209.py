#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z, show_plot=False):
    result = 1 / (1 + np.exp(-z))
    if show_plot:
        x = np.linspace(-10, 10, 1000)
        y = 1 / (1 + np.exp(-x))
        plt.plot(x, y, label='Sigmoid', color='blue')
        plt.title('Sigmoid Activation Function')
        plt.xlabel('z')
        plt.ylabel('sigmoid(z)')
        plt.legend()
        plt.show()
    return result
def tanh(z, show_plot=False):
    result = np.tanh(z)
    if show_plot:
        x = np.linspace(-10, 10, 1000)
        y = np.tanh(x)
        plt.plot(x, y, label='Tanh', color='green')
        plt.title('Tanh Activation Function')
        plt.xlabel('z')
        plt.ylabel('tanh(z)')
        plt.show()
    return result
def relu(z, show_plot=False):
    result = np.maximum(0, z)
    if show_plot:
        x = np.linspace(-10, 10, 1000)
        y = np.maximum(0, x)
        plt.plot(x, y, label='ReLU', color='red')
        plt.title('ReLU Activation Function')
        plt.xlabel('z')
        plt.ylabel('ReLU(z)')
        plt.legend()
        plt.show()
    return result
class Neuron:
    def __init__(self, weights, bias, activation='sigmoid'):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation = activation

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        if self.activation == 'sigmoid':
            print(f'Sigmoid Output = {sigmoid(z, show_plot=True)}')
        elif self.activation == 'tanh':
            print(f'Tanh Output = {tanh(z, show_plot=True)}')
        else:
            print(f'ReLU Output = {relu(z, show_plot=True)}')
inputs=np.array([0.5,-1.2,3.0])
weights=[0.4,-0.6,0.2]
bias = 0.1
for i in ['sigmoid', 'tanh', 'relu']:
    neuron = Neuron(weights, bias, activation=i)
    neuron.forward(inputs)


# In[ ]:




