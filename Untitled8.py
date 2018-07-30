
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r'C:\Users\jyothi\Downloads\WA_Fn-UseC_-HR-Employee-Attrition.csv')
df   


# In[26]:


df.isnull().count()
df.dtypes


# In[41]:


numerical = [u'Age', u'DailyRate', u'DistanceFromHome', u'Education', u'EmployeeNumber', u'EnvironmentSatisfaction',
       u'HourlyRate', u'JobInvolvement', u'JobLevel', u'JobSatisfaction',
       u'MonthlyIncome', u'MonthlyRate', u'NumCompaniesWorked',
       u'PercentSalaryHike', u'PerformanceRating', u'RelationshipSatisfaction',
       u'StockOptionLevel', u'TotalWorkingYears',
       u'TrainingTimesLastYear', u'WorkLifeBalance', u'YearsAtCompany',
       u'YearsInCurrentRole', u'YearsSinceLastPromotion',
       u'YearsWithCurrManager']
f=plt.figure(figsize=(15,15))
sns.heatmap(df[numerical].astype(float).corr().values)
plt.xlabel('columns')
plt.title('heat map correlation')


# In[68]:


categorical = []
for col, value in df.iteritems():
    if value.dtype == 'object':
        categorical.append(col)
categorical
num = df.columns.difference(categorical)
dnum=df[num]


# In[73]:


dfc=df[categorical]
dfc.drop(['Attrition'],axis=1)
catt=pd.get_dummies(dfc)
dfc=pd.concat([dnum,catt],axis=1)
dfc


# In[83]:


target_map = {'Yes':1, 'No':0}
target = df["Attrition"].apply(lambda x: target_map[x])
plt.bar(df['Attrition'].value_counts().index.values,df['Attrition'].value_counts().values)
plt.xlabel(df['Attrition'].value_counts().index.values)


# In[92]:


from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
X_train,x_test,Y_train,y_test=train_test_split(dfc,target,train_size=0.75)


# In[96]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

rf=RandomForestClassifier()
tr=rf.fit(X_train,Y_train)
pr=rf.predict(x_test)
accuracy_score(y_test, pr)


# In[112]:


f=plt.figure(figsize=(10,10))
plt.scatter(x=rf.feature_importances_,y=dfc.columns.values,c=rf.feature_importances_,marker='o')

