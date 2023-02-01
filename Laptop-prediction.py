#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pandas


# In[2]:


#pip install matplotlib


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv(r"C:\Users\HP\Desktop\ML - datasets\laptop_data.csv")
df


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.drop(columns=['Unnamed: 0'],inplace=True)


# In[9]:


df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')


# In[10]:


#converting into dtype int
df['Ram']=df['Ram'].astype('int32')
df['Weight']=df['Weight'].astype('float32')


# In[11]:


df.info()


# In[12]:


pip install seaborn


# In[13]:


import seaborn as sns


# In[14]:


sns.distplot(df['Price'])


# price col is skewed 

# In[15]:


df['Company'].value_counts().plot(kind='bar')


# In[16]:


sns.barplot(x=df['Company'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[17]:


df['TypeName'].value_counts().plot(kind='bar')


# In[18]:


sns.barplot(x=df['TypeName'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# notebook and netbook laptops are the cheapest whereas workstation and gaming laptops 
# fall in the higher price range

# In[19]:


df['ScreenResolution'].value_counts()


# checking whether touchscreen is affecting the price

# In[20]:


df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[21]:


df.head(20)


# In[22]:


sns.barplot(x=df['Touchscreen'], y=df['Price'])


# hence laptops with touchscreen are more expensive 

# In[23]:


df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[24]:


df


# In[25]:


sns.barplot(x= df['Ips'], y=df['Price'])


# In[26]:


new = df['ScreenResolution'].str.split('x',n=1,expand=True)
new


# In[27]:


df['x_res']= new[0]
df['y_res'] =new[1]


# In[28]:


df.head(10)


# In[29]:


df['x_res'] = df['x_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[30]:


df.head(5)


# In[31]:


df['x_res'] = df['x_res'].astype('int')
df['y_res'] = df['y_res'].astype('int')


# In[32]:


df.info()


# In[33]:


df.corr()['Price']


# In[34]:


df['ppi']= (((df['x_res']**2) + (df['y_res']**2) ** 0.5) / df['Inches']). astype(float)


# In[35]:


df.info()


# In[36]:


df


# In[37]:


df.corr()['Price']


# In[38]:


df.drop(columns=['ScreenResolution'], inplace=True)


# In[39]:


df.drop(columns=['x_res', 'y_res', 'Inches'],inplace=True)


# In[40]:


df


# In[41]:


df['Cpu'].value_counts()


# In[42]:


df['Cpu_Name']= df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
#splitting and converting it into string 


# In[43]:


df.head(5)


# #defining function to make seperate columns for intel, core, generation 

# In[44]:


def fetch_processor(text):
    if text=='Intel Core i7' or text=='Intel Core i5' or text== 'Intel Core i3':
        return text 
    else:
        if text.split()[0]=='Intel':
            return 'Other intel Processor'
        else:
            return 'AMD processor'


# In[45]:


df['Cpu_Brand']= df['Cpu_Name'].apply(fetch_processor)


# In[46]:


df.head()


# In[47]:


df['Cpu_Brand'].value_counts()


# In[48]:


df['Cpu_Brand'].value_counts().plot(kind='bar')


# In[49]:


sns.barplot(x=df['Cpu_Brand'] , y=df['Price'])
plt.xticks(rotation= 'vertical')
plt.show()


# In[50]:


df.drop(columns= ['Cpu','Cpu_Name'], inplace=True)


# In[51]:


df.head()


# In[52]:


df['Ram'].value_counts()


# In[53]:


sns.barplot(x= df['Ram'], y=df['Price'])


# varies linearly with price 

# In[54]:


df['Memory'].value_counts()


# In[55]:


newm=df['Memory'].str.split('x',n=1,expand=True)


# In[56]:


newm


# In[57]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[58]:


df.head(5)


# In[59]:


df.drop(columns=['Memory'],inplace=True)


# In[60]:


df.corr()['Price']


# In[61]:


#hybrid and flash storage dont conrtibute much to the price so we can drop them


# In[62]:


df.drop(columns=['Hybrid','Flash_Storage'], inplace=True)


# In[63]:


df.head()


# In[64]:


df['OpSys'].value_counts()


# In[65]:


sns.barplot(x= df['OpSys'], y= df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[66]:


df.head()


# In[67]:


def category_os(name):
    if name=='Windows 10 S' or name=='Windows 7'or name== 'Windows 10':
        return 'Windows'
    elif name=='macOS' or name== 'Mac OS X':
        return 'Mac'
    else:
        return 'Other/No OS/Linux'


# In[68]:


df['OS']= df['OpSys'].apply(category_os)


# In[69]:


df.head()


# In[70]:


df.drop(columns=['OpSys'],inplace=True)


# In[71]:


sns.barplot(x=df['OS'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[72]:


df['Gpu'].value_counts()


# In[73]:


df['Gpu_Brand']=df['Gpu'].apply(lambda x:x.split()[0])


# In[74]:


df = df[df['Gpu_Brand'] != 'ARM']


# In[75]:


df['Gpu_Brand'].value_counts()


# In[76]:


sns.barplot(x=df['Gpu_Brand'], y=df['Price'], estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[77]:


df.drop(columns=["Gpu"], inplace=True)


# In[78]:


df.head()


# In[79]:


sns.distplot(df['Weight'])


# In[80]:


sns.scatterplot(x= df['Weight'], y= df['Price'])


# In[81]:


df.corr()['Price']


# In[82]:


sns.heatmap(df.corr())


# In[83]:


sns.distplot(df['Price'])


# we can see that the plot is skewed

# In[84]:


sns.distplot(np.log(df['Price']))


# In[85]:


X= df.drop(columns=['Price'])
Y= np.log(df['Price'])


# In[86]:


X


# In[87]:


#conda install scikit-learn


# In[88]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X,Y,test_size=0.15, random_state=1)


# In[89]:


x_train


# In[90]:


x_test


# In[91]:


pip install xgboost


# In[92]:


from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[93]:


from sklearn.linear_model import LinearRegression ,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor, AdaBoostRegressor 
from sklearn.svm import SVR
from xgboost import XGBRegressor


# In[105]:


ohe= OneHotEncoder()
transformed= ohe.fit_transform(df[['Cpu_Brand']])
transformed.astype()


# ### Linear Regression 

# In[106]:


step1= ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop='first',handle_unknown='ignore'),[0,1,10,11])],
    remainder= 'passthrough')
    
step2= LinearRegression()
pipe= Pipeline([
              ('step1', step1),
              ('step2', step2)
])
    
pipe.fit(x_train, y_train)

y_pred= pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Decision Tree Regressor 

# In[95]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first',handle_unknown='ignore'),[0,1,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### SVM

# In[96]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first',handle_unknown='ignore'),[0,1,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Random Forest Regressor

# In[97]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first', handle_unknown='ignore'),[0,1,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=30)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### XGBRegressor

# In[98]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first',handle_unknown='ignore'),[0,1,10,11])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=52,max_depth=5,learning_rate=0.5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Exporting model

# In[99]:


import pickle 

pickle.dump(df, open('df.pckl', 'wb'))

pickle.dump(pipe, open('pipe.pckl', 'wb'))


# In[100]:


df


# In[101]:


df['Cpu_Brand']


# In[104]:


ohe= OneHotEncoder()
transformed= ohe.fit_transform(df[['Cpu_Brand']])
print(transformed.toarray())

