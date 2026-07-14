#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# In[52]:


df = pd.read_csv("../data/cookie_cats.csv")
df.head()

# Some business questions to be answered
# 
# ## Observed Groups
# 
# Control Group: Players encountered the first gate at level 30
# 
# Treatment Group: Players encountered the first gate at level 40
# 
# 
# ### What improves player retention more?
# 
# Placing the gate at an earlier date? (Level 30) bettter long-term retention
# 
# Placing the gate at a later date? (level 40) better retention

# In[3]:


df.isnull().sum()

# In[4]:


df.groupby('version')[['retention_1', 'retention_7']].value_counts(normalize="True").unstack()

# What happened in the sample, a quick descriptive analysis on each version

# In[5]:


categories = df['version'].unique()

def get_retention_count(retetion_value, retention_day):
    value_counts = []
    for category in categories:
        print(category)
        cat_count = df[(df['version'] == category) & (df[retention_day] == retetion_value)][retention_day].count()
        value_counts.append(cat_count)
    # return some array
    return value_counts

ret7_true = get_retention_count(True, 'retention_7')
print(ret7_true)

print(len(categories))

# In[29]:


categories = df['version'].unique()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

width=0.5

def get_retention_count(retetion_value, retention_day):
    value_counts = []
    for category in categories:
        df[(df['version'] == category) & (df[retention_day] == retetion_value)][retention_day].count()
        cat_count = df[(df['version'] == category) & (df[retention_day] == retetion_value)][retention_day].count()
        value_counts.append(cat_count)
    # return some array
    return value_counts

retention_proportions = {
    'retention_1_proprtions': {
        'Played': get_retention_count(True, 'retention_1'),
        'Did not play': get_retention_count(False, 'retention_1')
    },
    'retention_7_proprtions': {
        'Played': get_retention_count(True, 'retention_7'),
        'Did not play': get_retention_count(False, 'retention_7')
    }
}

i = 0
for _, retention_proportion in retention_proportions.items():
    bottom = np.zeros(len(categories))
    for boolean, ret_count in retention_proportion.items():
        p = axs[i].bar(categories, ret_count, width, label=boolean, bottom=bottom)
        bottom += ret_count
        
    i += 1

axs[0].set_title("Player retention within one day")
axs[1].set_title("Player retention within seven days")
axs[0].legend(loc="upper right")        
axs[1].legend(loc="upper right")        

# axs.set_title("Did player played for 7 days after installing")
# axs.legend(loc="upper right")

plt.show()

plt.show()

# How confident we are that the difference is real
# 
# ## For both retentions (Within 1 day and 7 days)
# 
# 1. Null hypotheses: Placing a gate at a different level does not improve player retention
# 
# 2. Alternative Hypotheses: Placing the gate at a different level would improve player retention

# In[7]:


from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

# First test for retention within one day
n_con = df[(df['version'] == 'gate_30')]
n_trt = df[(df['version'] == 'gate_40')]

# success
p_con = n_con[n_con['retention_1'] == True].shape[0]
p_trt = n_trt[n_trt['retention_1'] == True].shape[0]

count = [p_con, p_trt]
nobs = [n_con.shape[0], n_trt.shape[0]]


zstat, pval = proportions_ztest(count, nobs)
print('P Value: {0:0.3f}'.format(pval))
print('Z statistic: ', zstat)

# alpha
alpha = 0.05

# critical z value
z_critical_two_tailed = stats.norm.ppf(1 - alpha/2)

if (pval > alpha):
    print("We fail to reject the null hypotheses, so changing the level gate does not produce different 1 day retentions")
else:
    print("We reject the null hypotheses, indicating that there is a significant difference in 1 day retention when changing the gate placement")



# In[8]:


# First test for retention within seven days

# success
p_con = n_con[n_con['retention_7'] == True].shape[0]
p_trt = n_trt[n_trt['retention_7'] == True].shape[0]

count = [p_con, p_trt]
nobs = [n_con.shape[0], n_trt.shape[0]]


zstat, pval = proportions_ztest(count, nobs)
print('P Value: {0:0.3f}'.format(pval))
print('Z statistic: ', zstat)

# alpha
alpha = 0.05

# critical z value
z_critical_two_tailed = stats.norm.ppf(1 - alpha/2)

if (pval > alpha):
    print("We fail to reject the null hypotheses, so changing the level gate does not produce different 7 day retentions")
else:
    print("We reject the null hypotheses, indicating that there is a significant difference in 7 day retention when changing the gate placement")



# In[21]:


gate_30 = df[df['version'] == 'gate_30']
gate_40 = df[df['version'] == 'gate_40']

# Ratio Analysis
ratio_g30 = (gate_30.shape[0]/df.shape[0]) * 100
ratio_g40 = (gate_40.shape[0]/df.shape[0]) * 100

print(f'gate_30: {ratio_g30:.2f} %')
print(f'gate_40: {ratio_g40:.2f} %')


# We would also like to calculate if there is any significant difference between the mean of the total games play between the two groups.
# 
# We have no information about both population variance, so we use **Welch's T Test**

# In[53]:


from statsmodels.stats.weightstats import ttest_ind

# Samples gamerounds for each group
gamerounds_x = df[(df['version'] == 'gate_30')]['sum_gamerounds']
gamerounds_y = df[(df['version'] == 'gate_40')]['sum_gamerounds']

# Run Welch's T-Test
t_stat, p_value, degf = ttest_ind(gamerounds_x, gamerounds_y, usevar='unequal')

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"degrees of freedom: {degf:.3f}")


# Our dependent variable has a binary outcome
# 
# One of our goals: Model the probability of retention and estimate the treatment effect
# 
# Logistic Regression is the way to go: "Will the player keep playing the game after 7 days? Yes/No"

# In[ ]:



