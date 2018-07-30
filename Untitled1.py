
# coding: utf-8

# In[61]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import os
df=pd.read_excel(r'C:\Users\jyothi\Downloads\Sample - Superstore.xlsx')
df=df[['Order Date','Sales']].sort_values('Order Date',ascending=False)
df= df.rename(columns={'Order Date': 'od', 'Sales': 'sales'})
df.index=df.od
df=df[(df.od>'2018-01-01')]
df=df.drop('od',axis=1)
df
f=plt.figure(figsize=(20,5))
plt.plot(df)
plt.xlabel('Dates')
plt.ylabel('sales')
plt.title('sales by date')


# In[62]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df)


# In[67]:


from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(df,order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals =pd.DataFrame(model_fit.resid)
residuals.plot()

residuals.plot(kind='kde')
print(residuals.describe())


# In[69]:


X = df.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(10,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')

