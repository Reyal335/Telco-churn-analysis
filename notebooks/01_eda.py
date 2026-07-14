#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# In[3]:


df = pd.read_csv('../data/Telco_customer_churn.csv')
df

# In[62]:


df

# Data Inspection, Cleaning and Tranformation
# 

# In[63]:


# Dataframe basic information
df.info()

# In[64]:


df.describe()

# In[65]:


print('Dataframe Shape:')
print(df.shape)
print('')
print('Dataframe tail and head:')
print(df.head(10))
print(df.tail(5))
print('')
print('Dataframe info')
print(df.info())
print('')
print('Datafram types')
print(df.dtypes)
print('')
print(df.columns.tolist())

# Data Quality Assessment
# 

# In[66]:


# Missing Values
print(df.isnull().sum()[df.isnull().sum() > 0])

# In[91]:


df.loc[df['avg_monthly_spend'].isna(), ['Total Charges', 'Tenure Months']]

# In[68]:


import missingno as msno 
msno.matrix(df)
plt.show()

# ### Around 73% of Churn Reason is blank (NaN)

# In[69]:


df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# In[70]:


# Duplicated Values
print(df.duplicated().sum())

# In[71]:


df['Total Charges'].dtypes

# In[72]:


df.groupby('Contract').agg(
    total_charges=('Total Charges', 'sum')
)

# In[73]:


df['Churn Value'].value_counts(normalize='True')

# ## We have observed an overall 26% churn rate for this quarter
# 

# ### Churn Rate per contract type
# 

# In[1]:


df.groupby('Contract')['Churn Value'].value_counts(normalize='True')

# In[75]:


hist = df.hist(column='Contract', by='Churn Value')

# ### Churn Rate per Payment Method
# 

# In[5]:


df.groupby('Payment Method')['Churn Value'].value_counts(normalize='True')

# In[77]:


hist = df.hist(column='Payment Method', by='Churn Value')

# In[4]:


df.groupby('Tenure Months')['Churn Value'].value_counts(normalize='True')

# In[79]:


hist = df.hist(column='Tenure Months', by='Churn Value', bins=30, figsize=(10, 4))

# In[80]:


churned_customers = df[df['Churn Value'] == 1]
retained_customers = df[df['Churn Value'] == 0]

# sns.histplot(data=churned_customers, x='Tenure Months', bins=30, kde=True)

fig, ((ax0, ax1)) = plt.subplots(ncols=2, figsize=(10, 4))
fig.suptitle('Tenure Months Distribution for Churned and Retained Customers')

ax0.hist(churned_customers['Tenure Months'], bins=30)
ax1.hist(retained_customers['Tenure Months'], bins=30)

# In[81]:


hist = df.hist(column='Monthly Charges')

# ## Data Cleaning and Feature Engineering Phase

# In[82]:


# Convert Total Charges from string to float
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors='coerce')
df["Total Charges"].dtypes

# In[83]:


# Remove Entries that have NaN Total Charges
df.dropna(subset=['Total Charges'], inplace=True)

# In[84]:


# Convert Binary Yes/No columns encoded as 0/1

# df.head()
# Binary Yes/No columns
binary_cols = ['Gender', 'Paperless Billing', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service']
for col in binary_cols:
    print(col)
    map_key = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}
    df[col] = df[col].map(map_key)
    print(df[col])


# In[92]:


df.head()

# In[86]:


# One-hot encode multi-category column
df = pd.get_dummies(df, columns=['Contract', 'Internet Service', 'Payment Method'], drop_first=True)

# In[87]:


# Feature engineering
df['avg_monthly_spend'] = df['Total Charges'] / (df['Tenure Months'] + 1)
df['has_multiple_services'] = (
    df['Phone Service'] + df['Online Security'].map({'Yes':1,'No':0,'No internet service':0})
) > 1

# In[88]:


df.loc[df['Churn Value'] == 0, ['Churn Reason']]

# In[93]:


# Convert to CSV (Cleaned and ready for training)
df.to_csv('../data/Telc_customer_churn_cleaned.csv', index=False)

# In[ ]:



