#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# # Reading Data CSV File

# In[2]:


# Reading Data From CSV File
data = pd.read_csv('BlackFriday.csv')


# # Data Information

# In[3]:


data.info()


# # head Data Showing First 10 Rows From Data set..
# ## Here we can see that Product_Category_2 and Product_Category_3 has Null values

# In[4]:


# First 10 Data points (Rows)
data.head(10)


# # Matrix which describes features (numeric values data) 

# In[5]:


data.describe()


# # Data Cleansing

# ## Checking for null columns

# In[6]:


data.isnull().any()


# ## Null Columns

# In[7]:


null_columns = data.columns[data.isnull().any()]
print(null_columns)


# ## Filling Null values with 0

# In[8]:


data.fillna(value=0, inplace=True)
data


# # Exploratory Data Analysis (EDA)

# In[9]:


data['Product_Category_2'] = data['Product_Category_2'].astype('int64')
data['Product_Category_3'] = data['Product_Category_3'].astype('int64')
data.sort_values('User_ID')
data


# ## Gender Graph Without duplicates (User transactions)

# In[10]:


# Gender Pie Chart Without duplicates....
actualGender = data[['User_ID','Gender']].drop_duplicates('User_ID')
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(actualGender['Gender'].value_counts(),labels=['Male','Female'],autopct='%1.2f%%',explode = (0.1,0))
plt.title("Actual ratio between Male & Female")
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()


# In[11]:


fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(data[['Gender','Purchase']].groupby(by='Gender').sum(),labels=['Female','Male'],autopct='%1.2f%%',explode = (0.1,0))
plt.title("Ratio of purchase amount between male and female")
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()

print(data[['Gender','Purchase']].groupby(by='Gender').sum())


# # Married Customers 2474
# # Unmarried Customers 3417

# In[12]:


u = data['User_ID'].groupby(data['Marital_Status']).nunique()
u


# ## Stay_In_Current_City_Years
# ### 0      772
# ### 1     2086
# ### 2     1145
# ### 3      979
# ### 4+     909

# In[13]:


r = data['User_ID'].groupby(data['Stay_In_Current_City_Years']).nunique()
r


# In[14]:


print('Unmarried ', len(data[data['Marital_Status'] == 0]))


# In[15]:


print('married ', len(data[data['Marital_Status'] == 1]))


# In[16]:


AgePurchase_DF = data[['Age','Purchase']].groupby('Age').sum().reset_index()
print(type(AgePurchase_DF))
print(AgePurchase_DF)
agePurchase = data.groupby(['Age', 'Purchase']).sum()
# print(agePurchase)
# print(AgePurchase_DF)
fig1,ax1 = plt.subplots()
sns.barplot(x='Age',y='Purchase',data=AgePurchase_DF)
plt.title('Total purchase made by different Age')
plt.tight_layout()


# In[17]:


Customers = data['User_ID'].unique()
# for cust Customers:
    
# data.loc[data['User_ID'] == 1000001]


# In[18]:


print('Hello')


# In[19]:


gm = data.groupby('Marital_Status')


# In[20]:


# for i, d1 in gm:
#     print(i)
#     print(d1)


# In[21]:


# gm[0][0]


# In[22]:


# gm[0].Age


# In[23]:


mart = gm.get_group(0)


# In[24]:


# Customers with age (0-17) not married
mart_d = mart[mart['Age'] == '0-17']
len(mart_d['Age'])


# In[25]:


type(mart[mart['Age'] == '0-17'].count())


# In[26]:


mart[mart['Age'] == '0-17'].count().value_counts


# In[27]:


# Customers with age (18-25) not married
mart_d = mart[mart['Age'] == '18-25']
len(mart_d['Age'])


# In[28]:


# Customers with age (26-35) not married
mart_d = mart[mart['Age'] == '26-35']
len(mart_d['Age'])


# In[29]:


# Customers with age (36-45) not married
mart_d = mart[mart['Age'] == '36-45']
len(mart_d['Age'])


# In[30]:


# Customers with age (46-55) not married
mart_d = mart[mart['Age'] == '46-50']
len(mart_d['Age'])


# In[31]:


# Customers with age (51-55) not married
mart_d = mart[mart['Age'] == '51-55']
len(mart_d['Age'])


# In[32]:


# Customers with age (55+) not married
mart_d = mart[mart['Age'] == '55+']
len(mart_d['Age'])


# In[33]:


mr = data[data['Marital_Status'] == 0]
# mr
ageMarital_data = data[['Age','Marital_Status']].groupby('Marital_Status').count().reset_index()
# print(ageMarital_data.max())
print(ageMarital_data)
# ageMarital_data = mr[['Age','Marital_Status']].groupby('Age').count().reset_index()
# print(ageMarital_data)


# In[34]:


mr = data[data['Marital_Status'] == 0]
# mr
print('Age Groups who are unmarried')
ageMarital_data = mr[['Age','Marital_Status']].groupby('Age').count().reset_index()
print(ageMarital_data)
# print('Maximum Age Group UnMarried')
# print(ageMarital_data.max())
# print(ageMarital_data.describe())


# In[35]:


mr = data[data['Marital_Status'] == 1]
# mr
print('Age Groups who are married')
ageMarital_data = mr[['Age','Marital_Status']].groupby('Age').count().reset_index()
print(ageMarital_data)


# In[36]:


mr = data[data['Marital_Status'] == 1]
# mr
print('Age Groups who are married')
ageMarital_data = mr[['Age','Marital_Status']].groupby('Age').count().reset_index()
print(ageMarital_data)
# ageMarital_data.hist()


# In[37]:


g = mr[mr['Gender'] == 'F']
gage = g[['Gender', 'Age']].groupby('Age').count().reset_index()
gage


# In[38]:


g = mr[mr['Gender'] == 'M']
gage = g[['Gender', 'Age']].groupby('Age').count().reset_index()
gage


# In[39]:


mr


# In[40]:


data['Occupation'].unique()


# In[41]:


data['Stay_In_Current_City_Years'].unique()


# In[42]:


sns.countplot(data['Gender'])


# In[43]:


sns.countplot(data['Age'])


# In[44]:


sns.countplot(data['Age'], hue=data['Gender'])


# ## This graph makes sense under 18 are not married

# In[45]:


sns.countplot(data['Age'], hue=data['Marital_Status'])


# In[46]:


data['Stay_In_Current_City_Years'].unique()


# # Analysing User_ID Feature
# ## not enough information from here, customer are buying stuff/products irrespective of age, gender or city.
# ## But Male is buying more than female which is not common on further study it shows that 26-35 married group so may be husbands are paying bills

# In[47]:


fig, axes = plt.subplots(nrows=5, ncols=2,figsize=(15,15))
fig.tight_layout()
fig.subplots_adjust(wspace=0.4,hspace=0.4)

data['User_ID'].groupby(data['Gender']).nunique().plot(kind='bar',ax=axes[0,0])
data['User_ID'].groupby(data['Age']).nunique().plot(kind='bar',ax=axes[0,1])
data['User_ID'].groupby(data['Occupation']).nunique().plot(kind='bar',ax=axes[1,0])
data['User_ID'].groupby(data['City_Category']).nunique().plot(kind='bar',ax=axes[1,1])
data['User_ID'].groupby(data['Stay_In_Current_City_Years']).nunique().plot(kind='bar',ax=axes[2,0])
data['User_ID'].groupby(data['Marital_Status']).nunique().plot(subplots=True,kind='bar',ax=axes[2,1])
data['User_ID'].groupby(data['Product_Category_1']).nunique().plot(subplots=True,kind='bar',ax=axes[3,0])
data['User_ID'].groupby(data['Product_Category_2']).nunique().plot(subplots=True,kind='bar',ax=axes[3,1])
data['User_ID'].groupby(data['Product_Category_3']).nunique().plot(kind='bar',ax=axes[4,0])


# # Analysing Product_ID Feature
# ## Occupation 0 Customers bought more products

# In[48]:


fig, axes = plt.subplots(nrows=5, ncols=2,figsize=(16,15))
fig.tight_layout()
fig.subplots_adjust(wspace=0.4,hspace=0.4)

data['Product_ID'].groupby(data['Gender']).nunique().plot(kind='bar',ax=axes[0,0])
data['Product_ID'].groupby(data['Age']).nunique().plot(kind='bar',ax=axes[0,1])
data['Product_ID'].groupby(data['Occupation']).nunique().plot(kind='bar',ax=axes[1,0])
data['Product_ID'].groupby(data['City_Category']).nunique().plot(kind='bar',ax=axes[1,1])
data['Product_ID'].groupby(data['Stay_In_Current_City_Years']).nunique().plot(kind='bar',ax=axes[2,0])
data['Product_ID'].groupby(data['Marital_Status']).nunique().plot(subplots=True,kind='bar',ax=axes[2,1])
data['Product_ID'].groupby(data['Product_Category_1']).nunique().plot(subplots=True,kind='bar',ax=axes[3,0])
data['Product_ID'].groupby(data['Product_Category_2']).nunique().plot(subplots=True,kind='bar',ax=axes[3,1])
data['Product_ID'].groupby(data['Product_Category_3']).nunique().plot(kind='bar',ax=axes[4,0])


# In[49]:


print(data['Product_ID'].groupby(data['Product_Category_3']).nunique())


# In[50]:


data['Product_ID'].groupby(data['Product_Category_2']).unique().count()


# In[51]:


data['Product_ID'].groupby(data['Product_Category_1']).unique().count()


# In[52]:


fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(16,15))
fig.tight_layout()
fig.subplots_adjust(wspace=0.4,hspace=0.4)

data['Purchase'].groupby(data['Product_Category_1']).nunique().plot(kind='bar',ax=axes[0,0])
data['Purchase'].groupby(data['Product_Category_2']).nunique().plot(kind='bar',ax=axes[0,1])
data['Purchase'].groupby(data['Product_Category_3']).nunique().plot(kind='bar',ax=axes[1,0])


# In[53]:


age_product_gb = data[['Age', 'Product_ID', 'Gender', 'Purchase']].groupby(['Age', 'Product_ID', 'Gender']).agg('count')
age_product_gb.sort_values('Purchase', inplace=True, ascending=False)
ages = sorted(data.Age.unique())
result = pd.DataFrame({
    x: list(age_product_gb.loc[x].index)[:5] for x in ages
}, index=['#{}'.format(x) for x in range(1,6)])
result


# In[54]:


age_product_gb = data[['Age', 'Product_ID', 'Marital_Status', 'Purchase']].groupby(['Age', 'Product_ID', 'Marital_Status']).agg('count')
age_product_gb.sort_values('Purchase', inplace=True, ascending=False)
ages = sorted(data.Age.unique())
result = pd.DataFrame({
    x: list(age_product_gb.loc[x].index)[:5] for x in ages
}, index=['#{}'.format(x) for x in range(1,6)])
result


# In[55]:


age_product_gb = data[['Age', 'Product_ID', 'Marital_Status', 'Purchase']].groupby(['Age', 'Product_ID', 'Marital_Status'])
age_product_gb.first()


# In[56]:


data.head()


# In[57]:


newData = data
newData.head()


# # Encoding Categorical Variables
# ## just to show how encoding working created functions, it can be done with one line as well

# In[58]:


def map_gender(gender):
    if gender == 'F':
        return 0
    else:
        return 1
newData['Gender'] = newData['Gender'].apply(map_gender)


# In[59]:


def map_age(age):
    if age == '0-17':
        return 0
    elif age == '18-25':
        return 1
    elif age == '26-35':
        return 2
    elif age == '36-45':
        return 3
    elif age == '46-50':
        return 4
    elif age == '51-55':
        return 5
    else:
        return 6
newData['Age'] = newData['Age'].apply(map_age)


# In[60]:


newData['Age'].unique()


# In[61]:


def map_city_categories(city_category):
    if city_category == 'A':
        return 0
    elif city_category == 'B':
        return 1
    else:
        return 2
newData['City_Category'] = newData['City_Category'].apply(map_city_categories)


# In[62]:


newData['City_Category'].unique()


# In[63]:


def map_stay(stay):
        if stay == '4+':
            return 4
        else:
            return int(stay)
newData['Stay_In_Current_City_Years'] = newData['Stay_In_Current_City_Years'].apply(map_stay)   


# In[64]:


newData['Stay_In_Current_City_Years'].unique()


# # Correlation Matrix and Graph (Heat Map)
# ## Product_Category_1 and Product_Category_3 are reciprocal of each other negative correlation
# ## Product_Category_3 is important for puchase only that varible has good corr value for purchase
# ## Marital Staus and Age Correlation making our assumption thinking true

# In[65]:


# fig,ax = plt.subplots(figsize = (12,9))
# sns.heatmap(newData.drop(['User_ID','Product_ID'],axis=1).corr())
d1 = newData.drop(['User_ID', 'Product_ID'], axis=1)
# newData.head()
corr_matrix = d1.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, annot = True);
# sns.heatmap(corr_matrix, square=True);
plt.show()


# In[66]:


# del newData
d1


# In[67]:


print(d1['Age'].unique())
print(d1['Gender'].unique())
print(d1['Stay_In_Current_City_Years'].unique())


# In[68]:


from sklearn.model_selection import train_test_split
# X = d1.drop(['Purchase','Stay_In_Current_City_Years', 'Product_Category_1', 'Product_Category_2',
#              'Age', 'Marital_Status', 'Occupation', 'City_Category', 'Gender'],axis=1)
X = d1.drop(['Purchase', 'Product_Category_2'],axis=1)
# X = d1.drop(['Purchase'],axis=1)
X.info()
# df.reset_index()
Y = d1['Purchase']


# In[69]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[70]:


X_train['Product_Category_3'].values


# # Normalizing Data (0-1)

# In[71]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
nor = input('Normalize data ? press y for yes any letter for continue')
if(nor == 'y'):
    X_train = normalize(X_train)
    X_test  = normalize(X_test)


# In[72]:


print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
# X_train


# # Data Modeling

# ## Linear Regression Model

# In[92]:


def linear_regression(X_train, Y_train, X_test, Y_test, X, Y):
    print('.............Running Linear Regression Model............')
    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, Y_train)
    predictions = lm.predict(X_test)
    print('predicted values')
    print(predictions)
    print('actual values')
    print(np.array(d1.iloc[430062:537577].Purchase))
    print('Training data set score', model.score(X_train, Y_train))
    print('Test data set score', model.score(X_test, Y_test))
    yn = input("Want to apply KFold? press y for yes any letter to continue ")
    if(yn == 'y'):
        scores = cross_val_score(model, X, Y, cv=3)
        print("Cross-validated scores:", scores)
        print(scores.max())


# ## Training and test score 
# ## Even after normalizing the data set accuracy not increased

# ## Cross validation (Kfold by cross_val_score)

# # Cross validation too not given great score
# ## Alternate Solution for that problem is RandomForest Model as normalizing encoding not worked

# ## RandomForest 

# In[94]:


def random_forest(X_train, Y_train, X_test, Y_test, X, Y):
    print('.............Running RandomForest Model............')
    regr = RandomForestRegressor(max_depth=11,random_state=42,n_estimators=150)
    regr.fit(X_train,Y_train)
    rtest_score = regr.score(X_test,Y_test)
    rtrain_score = regr.score(X_train,Y_train)
    print('Test data set Score: ', rtest_score)
    print('Train data set Score: ', rtrain_score)
    yn = input("Want to apply KFold? press y for yes any letter to continue ")
    if(yn == 'y'):
        scores = cross_val_score(regr, X, Y, cv=3)
        print(scores)


# ## Cross validation (Kfold by cross_val_score)

# # RandomForest Performing quite well than linear regression almost 70% accuracy reported, by changing some parameters

# # KNN Algorithm for gender prediction with purchase and occupation

# In[91]:


def knn():
    print('.............Running KNN Clasifier Model............')
    knn = KNeighborsClassifier(n_neighbors = 3)
    ds = data.copy()
    ds1 = ds[['Gender', 'Occupation', 'Purchase']]
    X1,Y1 = ds1.loc[:,ds1.columns != 'Gender'], ds1.loc[:,'Gender']
    X_train1,X_test1,Y_train1,Y_test1 = train_test_split(X1,Y1,test_size = 0.3, random_state = 5)
    knn.fit(X_train1,Y_train1)
    prediction = knn.predict(X_test1)
    print(Y_test1.unique())
    print(prediction)
    print('knn Test Data Score: ', knn.score(X_test1,Y_test1))
    print('knn Training Score: ', knn.score(X_train1, Y_train1))
    yn = input('you want check best K ? press y for yes any letter to continue ')
    if(yn == '1'):
        n = np.arange(1,30)
        train_accuracy = []
        test_accuracy = []
        inertias = []
        for i, k in enumerate (n):
            knn = KNeighborsClassifier(n_neighbors = k)
            knn.fit(X_train1,Y_train1)
            train_accuracy.append(knn.score(X_train1,Y_train1))
            test_accuracy.append(knn.score(X_test1,Y_test1))

        # Plot
        plt.figure(figsize=(13,8))
        plt.plot(n, test_accuracy, label = 'Testing Accuracy')
        plt.plot(n, train_accuracy, label = 'Training Accuracy')
        plt.legend()
        plt.title('-value vs. Accuracy')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Accuracy')
        plt.xticks(n)
        plt.show()
        i = 1 + test_accuracy.index(np.max(test_accuracy))
        print('Best Accuracy is {} with K = {}'.format(np.max(test_accuracy),i))
        print('Training Accuray: {}'.format(train_accuracy[i]))


# # KNN Accuracy (K = 3)
# ## Training Data Set 78
# ## Testing Data Set 70
# ## KNN performing good

# ## Searching for Best K with range (1, n)

# # Best Test Accuracy is 0.751361037737019 with K = 27 (75.1 percent)
# # Accuracy of Training set  is 0.75616 (75.6 percent)

# In[93]:


q = 'n'
while q != 'y':
    n = input('Enter choice: press 1 for linear, 2 for random forest any for knn ')
    if(n == '1'):
        linear_regression(X_train, Y_train, X_test, Y_test, X, Y)
    elif(n == '2'):
        random_forest(X_train, Y_train, X_test, Y_test, X, Y)
    else:
        knn()
    q = input('You want to exit ? press y for yes any letter to continue... ')


# In[ ]:


# linear_regression(X_train, Y_train, X_test, Y_test, X, Y)
# random_forest(X_train, Y_train, X_test, Y_test, X, Y)
# knn()

