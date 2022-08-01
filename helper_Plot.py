#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from IPython import display
import pandas as pd 
import numpy as np 
display.set_matplotlib_formats('svg')

plt.rcParams['text.color'] = 'w'
plt.rcParams['xtick.color'] = 'w'
plt.rcParams['ytick.color'] = 'w'
plt.rcParams['axes.labelcolor'] = 'w'
plt.rcParams['legend.facecolor'] = 'k'


# In[2]:


def plot_single_graph( *data, xlabel='x', ylabel='y', title='', **params):
    '''
    eg, 
    params={'marker':'o','markerfacecolor':'g', 'linewidth':0.0}
    x=[0.1,0.5,0.8]
    y=[3,6,9]
    plot_single_graph(x,y, xlabel='x', ylabel='y', title='',**params)
    '''
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)    
    plt.plot(*data,**params)

    plt.show()


# In[3]:





# In[ ]:




