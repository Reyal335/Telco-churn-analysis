#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


x = np.arange(0, 10, 0.2)
y = np.sin(x)

x

# In[10]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)

# In[35]:


import seaborn as sns

sns.kdeplot(x)

# In[29]:


x = np.random.randn(500)
plt.plot(stats.cumfreq(x, numbins=10)[0])
print(stats.cumfreq(x, numbins=10)[0])
print(np.max(x))

# In[3]:


index = np.arange(5)
y = index**2
errorBar = index/2
print(errorBar)
plt.errorbar(index,y, xerr=errorBar, yerr=errorBar, fmt='o', capsize=5,capthick=3)

# In[5]:


x = np.random.randn(300)
plt.boxplot(x, '*')

# In[9]:


import pandas as pd

# In[14]:


nd = stats.norm
data = nd.rvs(size=(100))

nd2 = stats.norm(loc=3, scale=1.5)
data2 = nd2.rvs(size=(100))

df = pd.DataFrame({'Boy':data2, 'Girl': data})
sns.violinplot(df)
