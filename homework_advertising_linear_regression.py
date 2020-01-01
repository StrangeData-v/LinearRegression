#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[26]:


import csv
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split as ttsplit
import pandas as pd

#import the data using py.csv package
data_list = []

with open('C:\\Users\\13115\\epy\\aaa.csv') as fr:
    cr = csv.reader(fr)
    for each in cr:
        data_list.append(each)

dl = []

for a in data_list[1:]:
    dl.append([float(i) for i in a])
    
fd = np.array(dl)

x,y = fd[:,1:4],fd[:,4]
lr = LinearRegression()
x_train,t_test,y_train,y_test = ttsplit(x,y,test_size = 0.2,random_state = 2)
lr.fit(x_train,y_train)

print('coef:',lr.coef_)
print('intercept:',lr.intercept_)

(lr.predict(x) - y).mean()/y.mean()


# In[ ]:




