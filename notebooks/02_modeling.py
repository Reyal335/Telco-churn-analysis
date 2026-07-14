#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[29]:


df = pd.read_csv('../data/Telc_customer_churn_cleaned.csv')

# Drop the target and ANY other column that implicitly leaks the target
leakage_cols = ['Churn Value', 'CustomerID', 'Churn Score', 'Satisfaction Score']
X = df.drop(leakage_cols, axis=1, errors='ignore')
y = df['Churn Value']

df

# In[33]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# In[34]:


# Select only numerical columns for scaling
X_train_num = X_train.select_dtypes(include=['number'])
X_test_num = X_test.select_dtypes(include=['number'])

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_num)
X_test_sc = scaler.transform(X_test_num)


# In[37]:


# Baseline: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sc, y_train)
print("LR AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:,1]))

# In[41]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_num, y_train)
print("RF AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_num)[:,1]))
print(classification_report(y_test, rf.predict(X_test_num)))

# In[42]:


# 1. Print Random Forest feature importances
importances = pd.Series(rf.feature_importances_, index=X_train_num.columns)
print("--- TOP 5 LEAKING FEATURES ---")
print(importances.sort_values(ascending=False).head(5))

print("\n--- CORRELATION WITH TARGET ---")
# 2. Check direct linear correlation with Churn Value
correlations = X_train_num.corrwith(y_train).abs()
print(correlations.sort_values(ascending=False).head(5))

# In[45]:


top10 = importances.nlargest(10)
print(top10)

# In[48]:


top10.sort_values().plot(kind='barh', figsize=(8,5),
    color='#7F77DD', title='Top 10 churn drivers')
plt.tight_layout()
