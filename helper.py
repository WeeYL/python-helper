#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import inspect
from IPython import display
display.set_matplotlib_formats('svg')
import matplotlib.pyplot as plt

from IPython.display import display as ppi

def pp (*a):
    for i,n in enumerate(a): 
        i+=1
        if len(str(i))==1:z="00"
        elif len(str(i))==2:z="0"
        else: z=""
        print(n)
        print("-"*20,f"{z}{i}")

def ppn(key_li, val_li,delimiter=":"):
    '''input {list of key and value} '''
    li = []
    for k,v in zip(key_li,val_li):
        li.extend([str(k)," ",delimiter," ",str(v),"\n"])
    res = "".join(li)
    print (res)
    print('-'*20)

def pps(func):
    '''print source code'''
    print(*inspect.getsourcelines(func)[0])


# In[75]:


# import numpy as np 
# arr = np.arange(1,1200)
# pp(*arr)


# In[ ]:




