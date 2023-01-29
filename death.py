# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:21:32 2022

Leen Alnajjar

Life expectancy
"""
import numpy as np
import pandas as pd
filename='C:\\Users\\hp\\Desktop\\Personal files\\HTU\\python\\Machine learning\\Regression\\multiple reg\\life expectancy\\deathrate.csv'
death=pd.read_csv(filename,skipinitialspace = True)

#use info to get general information:
info=death.info()
#check mixed data type:`
from pandas.api.types import infer_dtype
mixed=death.apply(lambda x: 'mixed' in infer_dtype(x))

#check for duplicted values:
duplicated=death.duplicated().sum() #no duplicated rows

#Check for null values:
null=death.isnull().sum() # The total number of null values is 2563
#Deal with data problems:
#Group by year:
# applying groupby() function to
# group the data on team value in order to replace the missing value:
#for missing values:
#ensure that each country has its own mean:
#I'm afraid you cannot use only SimpleImputer for this kind of problem (at least as far as I know).
mean_replacement=death.groupby('Country').mean()
# Now replace based on the country:
death = death.fillna(death.groupby(['Country']).transform('mean')) # The null values now were reduced to 1698

#Check if there is a difference between the developed and developing countries in mean:

status_check=death.loc[:, death.columns != 'Year'].groupby('Status').mean()
#Now replce the remain null values based on the status:
    
death = death.fillna(death.groupby(['Status']).transform('mean')) # The null values now were reduced to zero


#The following step is to deal with the caterogical daata:
#The selected method to dael with the catero
#When trying to use the labeling method an error appear regarding the header name, to check that use tolist col:
whitespaces=death.columns.tolist() #Check from jupyter it will be clearer

#Now remove all white spaces:
    
death=death.rename(columns=lambda x: x.strip())

#You can now deal with the categorical data:
#Deal with status:
Unique=death.iloc[:,:].nunique() #this important for label encoder
from sklearn.preprocessing import LabelEncoder #Library imported
labelencoder=LabelEncoder() #Object defined
death['Status']=labelencoder.fit_transform(death['Status'])

#Deal with the contry:
mean = death.groupby('Country')['Life expectancy'].mean()
death['Country'] = death['Country'].map(mean)
'''
from category_encoders import TargetEncoder

targetencoder = TargetEncoder()
X[:,0] = encoder.fit_transform(X[:,0], y) #You must pip install
                                          #category_encoders library.
'''                                          

#Now split the data before continue:
    
X=death.loc[:, death.columns != 'Life expectancy'].iloc[:,:].values # Independent variables  
y=death.iloc[:,3].values #Dependent variables

#You can also split the data using the below method:

'''
d1 = death.iloc[:,:3].values
d2 = death.iloc[:,4:].values

df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)

X = pd.concat([df1, df2], axis=1)
y=death.iloc[:,3].values

'''
#OR:
'''
X = dataset.loc[:,dataset.columns.difference(['Life expectancy '],sort=False)].values
y = dataset.iloc[:,19:20].values
#Deal with the year:
'''
'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columntransformer=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder= 'passthrough')
X=columntransformer.fit_transform(X)
X=X[:,1:] # To remove the first col
'''
#OR you can use label encoder for better performance:
''
#Deal with the year:
labelencoder=LabelEncoder() #Object defined
death['Year']=labelencoder.fit_transform(death['Year'])


#The data now is ready for test and train:
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Use linear regression:
from sklearn.linear_model import LinearRegression

linearregression = LinearRegression()

linearregression.fit(X_train,y_train)

#Define the Y for prediction values
y_pred = linearregression.predict(X_test)

#Test the accuracy:
import sklearn.metrics as acc
print("Mean absolute error =", round(acc.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean squared error =", round(acc.mean_squared_error(y_test, y_pred), 2)) 
print("Median absolute error =", round(acc.median_absolute_error(y_test, y_pred), 2)) 
print("Explain variance score =", round(acc.explained_variance_score(y_test, y_pred), 2)) 
print("R2 score =", round(acc.r2_score(y_test, y_pred), 2))
#Now apply the multiple regression and retest the accuracy:
#Test the correlation to ensure the droped col:
#correlation to target values:
corr_matrix=death.corr()
coef=corr_matrix["Life expectancy"].sort_values(ascending=False)
#correlation to positive values:
pos_corr=coef[coef>0]
#correlation to negative values:
neg_corr=coef[coef<0]

X=np.append(arr=np.ones((2938,1)),values=X,axis=1)


import statsmodels.api as sm

def reg_ols(X,y):
    columns=list(range(X.shape[1]))#it will do it just one time
    a={}
    for i in range(X.shape[1]):
        X_opt=np.array(X[:,columns],dtype=float) #every time X_opt will change depend on the columns
        regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()#regressor between x_opt and y
        pvalues = list(regressor_ols.pvalues)#save pvalues column as a list
        d=max(pvalues)#find the maximum value in the list
        if (d>0.06):#if the maximum value bigger than 0.09 then we need to drop the column
            for k in range(len(pvalues)):#for loop to check the values in pvalues to find where is the max
                if(pvalues[k] == d):#if the value of in the index = max the delete it
                    a[k]=d
                    del(columns[k])  
    
    return(X_opt,regressor_ols,a)

X_opt,regressor_ols,a=reg_ols(X, y)
regressor_ols.summary()

#splitting the dataset
from sklearn.model_selection import train_test_split
X_opt_train,X_opt_test,y_opt_train,y_opt_test=train_test_split(X_opt,y,test_size=1/3,random_state=0)

#Training the Simple Linear Reg Model on the Training set
from sklearn.linear_model import LinearRegression
linearRegression2=LinearRegression()
linearRegression2.fit(X_opt_train,y_opt_train)

y_opt_pred=linearRegression2.predict(X_opt_test)


print('After applying MLR: ')
print("Mean absolute error =", round(acc.mean_absolute_error(y_opt_test,y_opt_pred), 2)) 
print("Mean squared error =", round(acc.mean_squared_error(y_opt_test,y_opt_pred), 2)) 
print("Median absolute error =", round(acc.median_absolute_error(y_opt_test,y_opt_pred), 2)) 
print("Explain variance score =", round(acc.explained_variance_score(y_opt_test,y_opt_pred), 2))
print("R2 score =", round(acc.r2_score(y_opt_test,y_opt_pred), 2))   
'''
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4)
X_poly = poly_reg.fit_transform(X)
linearRegression3 = LinearRegression()
linearRegression3.fit(X_poly, y_train)
X_poly_train,X_poly_test,y_poly_train,y_poly_test=train_test_split(X_poly,y,test_size=1/3,random_state=0)
y_pred2 = linearRegression3.predict(X_test)

print('After polyreg: ')
print("Mean absolute error =", round(acc.mean_absolute_error(y_poly_train,y_pred2), 2)) 
print("Mean squared error =", round(acc.mean_squared_error(y_poly_train,y_pred2), 2)) 
print("Median absolute error =", round(acc.median_absolute_error(y_poly_train,y_pred2), 2)) 
print("Explain variance score =", round(acc.explained_variance_score(y_poly_train,y_pred2), 2))
print("R2 score =", round(acc.r2_score(y_poly_train,y_pred2), 2))   
'''
from sklearn.ensemble import RandomForestRegressor
regressor = (RandomForestRegressor(n_estimators = 40, random_state = 0))
regressor.fit(X_train, y_train)
y_pred_RF=regressor.predict(X_test)
print('Accuracy of RandomForestRegressor= ',acc.r2_score(y_test, y_pred_RF))

from sklearn.tree import DecisionTreeRegressor
DTR = (DecisionTreeRegressor(random_state = 0))
DTR.fit(X_train, y_train)
y_pred_DTR=DTR.predict(X_test)
print('Accuracy of DecisionTreeRegressor= ',acc.r2_score(y_test, y_pred_DTR))
#Try the Ridge algorithm:
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
X_train_sc=standardscaler.fit_transform(X_train)
X_test_sc=standardscaler.fit_transform(X_test)  

ridge=Ridge()
ridge.fit(X_train_sc,y_train)
y_pred_ridge=ridge.predict(X_test_sc)
print('Accuracy of Ridge= ',acc.r2_score(y_test, y_pred_ridge))
print("Accuracy of Multiregression =",acc.r2_score(y_opt_test,y_opt_pred)) 

# Support Vector regressor algorithm:
'''
from sklearn.svm import SVR
SV=(SVR(kernel = 'linear')) #in this case linear has the highest accuracy
SV.fit(X_train,y_train)
y_pred_SV=SV.predict(X_test_sc)
print('Accuracy of SVR= ',acc.r2_score(y_test, y_pred_SV)) 

'''
'''
#Compare the accuracy:
print('Accuracy of RandomForestRegressor= ',acc.r2_score(y_test, y_pred_RF))
print('Accuracy of DecisionTreeRegressor= ',acc.r2_score(y_test, y_pred_DTR))    
print('Accuracy of SVR= ',acc.r2_score(y_test, y_pred_SV))
print("Accuracy of Multiregression =",acc.r2_score(y_opt_test,y_opt_pred))
print('Accuracy of Ridge= ',acc.r2_score(y_test, y_pred_ridge))

'''
