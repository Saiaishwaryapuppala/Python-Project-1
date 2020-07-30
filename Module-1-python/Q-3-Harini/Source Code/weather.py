#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

#Reading Dataset
data = pd.read_csv('weather.csv')

#Exploring the Data
data.head()
data.columns
data.shape
data.describe()
data.info()

#checking null values
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns  = ['Null Count']
nulls.index.name  = 'Feature'
print(nulls)

#counting null values
count=data['Precip Type'].value_counts()
print(count)

# Replacing the null values of 'Precip type' with 'rain'
data.loc[data['Precip Type'].isnull(), 'Precip Type'] = 'rain'

#checkin  null values
null=data.isnull().sum()
print(null)

# Performing label encoding for categorical values
data.loc[data['Precip Type'] == 'rain', 'Precip Type'] = 0
data.loc[data['Precip Type'] == 'snow', 'Precip Type'] = 1

# Dropping the unnecessary columns "Formmated Data", Summary" & "Daily Summary"
data.drop(['Summary'], axis=1, inplace=True)
data.drop(['Daily Summary'], axis=1, inplace= True)
data.drop(['Formatted Date'], axis=1, inplace= True)
data.drop(['Loud Cover'], axis=1, inplace= True)
data.head()

# Dividing the features and target
X = data.drop(['Temperature (C)'], axis=1)
y = data['Temperature (C)']

y.head()

# splitting data into test and train data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

X_train.head()

y_train.head()

# Implementing LinearRegression model
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

#predicted values

plt.scatter(X_test['Humidity'],predictions)
plt.show()
plt.scatter(X_test['Precip Type'],predictions)
plt.show()
plt.scatter(X_test['Apparent Temperature (C)'],predictions)
plt.show()

#