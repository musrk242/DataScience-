
# coding: utf-8

# In[57]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from  sklearn import datasets
iris=sns.load_dataset("iris")
df=iris
df=pd.DataFrame(df)
f,ax=plt.figure(figsize=(18,18))
ax1=f.add_subplot(2,2,1)
ax1.violinplot(df["petal_length"])
ax2=f.add_subplot(2,2,2)
sns.violinplot(x="species",y="petal_length",data=iris,ax=ax2)


# In[32]:


from sklearn import datasets
import pandas as pd
iris=datasets.load_iris()
df=iris.data
df=pd.DataFrame(df)
df


# In[58]:


# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted

# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')


# lets convert object(str) to categorical and int to float.
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')
data.info()
data["Type 2"].value_counts(dropna =False)
# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Type 2"].dropna(inplace = True)
data["Type 2"].fillna('empty',inplace = True)

country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df

data1.plot(subplots = True)
plt.show()
data.HP.apply(lambda n : n/2)

# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data3 then change index 
data3 = data.copy()
# lets make index start from 100. It is not remarkable change but it is just example
data3.index = range(100,900,1)
data3.head()

# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section
# It was like this
# data= data.set_index("#")
# also you can use 
# data.index = data["#"]

df.groupby("treatment").mean() 
df.groupby("treatment")[["age","response"]].min() 

f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('graph.png')
plt.show()

