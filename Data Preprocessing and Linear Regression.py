#!/usr/bin/env python
# coding: utf-8

# In[41]:


#Basic data Preprocessing and Linear Regression !


# In[43]:


import numpy as np #numpy for numerical computing
import pandas as pd #pandas for analysis
import seaborn as sns #seaborn for statistical graphs
import matplotlib.pyplot as plt #for plotting
get_ipython().run_line_magic('matplotlib', 'inline #making sure we can display our plots.')


# In[45]:


ecomcustomers=pd.read_csv('/Users/adkeme/Downloads/Ecommerce Customers') #import our dataset
ecomcustomers #print dataset


# In[46]:


ecomcustomers.info #basic info on dataset, ecomcustomers.head() or .tail() would be similar


# In[47]:


ecomcustomers.describe() #helpful to see # off columns, averages, mins etc


# In[50]:


sns.heatmap(ecomcustomers.isnull())
#We created a heatmap of our dataset using its null values. Our dataset is not missing any values.


# In[53]:


ecomcustomers.isna().sum() #Another way to make sure no missing values for 100% sure.


# In[54]:


sns.pairplot(data=ecomcustomers)
#pairplot will show us the relationship between all of our columns using scatterplots.


# In[57]:


sns.rugplot(ecomcustomers['Yearly Amount Spent'])
#Similar to our plot from earlier we see that on average $400 to $600 is the average spent.


# In[58]:


ecomcustomers['Yearly Amount Spent'].mean()
#the exact precise mean for the yearly amount is $499.


# In[59]:


ecomcustomers
#We will now begin deciding on what columns we need and dont need for our model.
#Email, Address and Avatar are specific and dont help, so we will drop them.


# In[60]:


ecomcustomers.drop(['Email','Address','Avatar'],inplace=True,axis=1)
#we drop the 3 columns, inplace is to make it permanent, axis is because its a column, not row.


# In[61]:


ecomcustomers


# In[69]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[70]:


ecomcustomers.columns


# In[76]:


X=ecomcustomers[['Avg. Session Length', 'Time on App', 'Time on Website',
       'Length of Membership']]
y = ecomcustomers['Yearly Amount Spent']
#we split our variables into X and Y's with X's being our independent and Y our dependent


# In[77]:


lm=LinearRegression() #create an instance for Linear Regression


# In[79]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)
#we set our parameters


# In[81]:


lm.fit(X_train,y_train)
#We train our model throug the created instance


# In[82]:


print('Coefficents:',lm.coef_)
#Print our coefficents


# In[84]:


predictions=lm.predict(X_test)
#store our predictions


# In[85]:


predictions


# In[86]:


plt.scatter(y_test,predictions)
plt.xlabel('Y test')
plt.ylabel('Predictions')
#Lets visualize our results through our graph.


# In[87]:


coefficents=pd.DataFrame(lm.coef_,X.columns)
coefficents.columns=['Coefficent']
coefficents


# In[88]:


#1 unit increases in each row equates the corresponding dollar value.
#the Length of membership and time on app have the strongest correspondence.


# In[89]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[91]:


print('Our MSE is:',mean_squared_error(y_test,predictions))
print('Our MAE is:',mean_absolute_error(y_test,predictions))
print('Our r2_score is:',r2_score(y_test,predictions))


# In[ ]:




