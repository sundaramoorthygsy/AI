#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
x1=[10,20,30]
x2=[15,25,35]
y=[50,60,70]
n=len(y)

random_init = np.random.randn(3)
w1 = random_init[0]
w2 = random_init[1]
b = random_init[2]

x1,x2,y,n,w1,w2,b


# In[37]:



def forward_pass(w1, w2, b):
    y_pred =list()
    y_pred =list()
    total_error = 0
    for i in range(n):
        y_hat = w1* x1[i] + w2* x2[i] + b
        y_pred.append(y_hat)

        E = (y[i] - y_hat)**2
        total_error+=E
    return y_pred, total_error
y_pred ,total_error


# In[38]:


def weight_update(w1, w2, b, y1):
    for i in range(n):
        de_dw1 = -2 * x1[i] * (y[i]-y_pred[i])
        de_dw2 = -2 * x2[i] * (y[i]-y_pred[i])
        de_db = -2 * (y[i]-y_pred[i])

        w1 = -alpha * de_dw1
        w2 = -alpha * de_dw2
        b = -alpha * de_db
    return w1, w2, b
w1,w2,b


# In[39]:


alpha = 0.0005
epochs = 10
error = list()
for epoch in range(epochs):
    y_pred, total_error = forward_pass(w1, w2, b)
    total_error1=round(total_error , 2)
    print(f"Epoch#:{epoch} | Error = {total_error1}")
    error.append(total_error)
    w1, w2, b = weight_update(w1, w2, b, y_pred)


# In[ ]:





# In[ ]:




