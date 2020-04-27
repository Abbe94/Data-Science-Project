# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:09:19 2020

@author: Abraham
"""


# Import all the packages that is needed for the project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
#import seaborn as sns
from scipy import stats
#from random import sample 


#from sklearn import datasets, linear_model
#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#import sklearn as sk
from statsmodels.nonparametric.smoothers_lowess import lowess
#from matplotlib import rcParams
import statsmodels.formula.api as sm
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso





# First, import the dataset as an numpy array named autompgdata
df=autompgdata[:,0:8]
data=pd.DataFrame(data=df)
n,p=data.shape 
#data.head(3)
#data.tail(3)
data.columns=["mpg","cylinders","displacement","horsepower","weight","acceleration","modelyear","origin"]
# data.dtypes # Find the datatype of each variable in the dataframe
data1 = data.apply(pd.to_numeric, errors='coerce')
ind=np.arange(1, n+1)

def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)

data1.horsepower[which(data.horsepower!="?")].describe() # Summary Statistics of Horsepower
data1.horsepower[which(data.horsepower=="?")]=round(data1.horsepower[which(data.horsepower!="?")].mean())
data2=data1.set_index(ind)







# Histograms
plt.hist(data2['horsepower'], density=True, bins=25, color='blue', edgecolor='black')
#plt.hist(data2['cylinders'], density=True, bins=25, color='blue', edgecolor='black')
plt.hist(data2['mpg'], density=True, bins=20, color='blue', edgecolor='black')
plt.hist(data2['displacement'], density=True, bins=50, color='blue', edgecolor='black')
plt.hist(data2['weight'], density=True, bins=30, color='blue', edgecolor='black')
plt.hist(data2['acceleration'], density=True, bins=30, color='blue', edgecolor='black')






# Scatter plots
data2.plot(kind='scatter', y='mpg', x='displacement', color='yellow')
data2.plot(kind='scatter', y='mpg', x='weight')
data2.plot(kind='scatter', y='mpg', x='horsepower')
data2.plot(kind='scatter', y='mpg', x='acceleration')










# Counter is the same as table in R
Counter(data1['cylinders'])
Counter(data1['origin'])
Counter(data1['modelyear'])






# Dealing with the the variables cylinders, origin and modelyear
data2=pd.get_dummies(data=data2, columns=['cylinders', 'origin', 'modelyear'])











# Splitting dataset into training data and testing data
y=data2.mpg
data2=data2.drop('mpg', axis=1)
X_train, X_test, y_train, y_test = train_test_split(data2, y, test_size=98/398)
correlationmatrix=data1.corr()
#ii=np.sort(np.array(sample(range(1,n),300)))
#traindata=data2.loc[ii]
#traindata=traindata.set_index(ii)
#nii,pii=traindata.shape
#ind1=set(ind)
#ind2=set(ii)
#ind1.difference_update(ind2)
#ind1=list(ind1)
#ind1=np.array(ind1)
#testdata=data2.loc[ind1]
#testdata=testdata.set_index(ind1)








# Linear regression
model=sm.OLS(y_train, X_train)
results=model.fit()
print(results.summary())









# Diagnostic plots: First residuals vs fitted values plot
residuals_OLS = results.resid
fitted_OLS = results.fittedvalues
smoothed = lowess(residuals_OLS,fitted_OLS)
top3 = abs(residuals_OLS).sort_values(ascending = False)[:3]

plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (8,7)
fig, ax = plt.subplots()
ax.scatter(fitted_OLS, residuals_OLS, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('Residuals')
ax.set_xlabel('Fitted Values')
ax.set_title('Residuals vs. Fitted')
ax.plot([min(fitted_OLS),max(fitted_OLS)],[0,0],color = 'k',linestyle = ':', alpha = .3)
for i in top3.index:
    ax.annotate(i,xy=(fitted_OLS[i],residuals_OLS[i]))
plt.show()







# 2nd diagnostic plot: QQ-plot
sorted_student_residuals_OLS = pd.Series(results.get_influence().resid_studentized_internal)
sorted_student_residuals_OLS.index = results.resid.index
sorted_student_residuals_OLS = sorted_student_residuals_OLS.sort_values(ascending = True)
df_OLS = pd.DataFrame(sorted_student_residuals_OLS)
df_OLS.columns = ['sorted_student_residuals']
df_OLS['theoretical_quantiles'] = stats.probplot(df_OLS['sorted_student_residuals'], dist = 'norm', fit = False)[0]
rankings = abs(df_OLS['sorted_student_residuals']).sort_values(ascending = False)
top3 = rankings[:3]

fig, ax = plt.subplots()
x = df_OLS['theoretical_quantiles']
y = df_OLS['sorted_student_residuals']
ax.scatter(x,y, edgecolor = 'k',facecolor = 'none')
ax.set_title('Normal Q-Q')
ax.set_ylabel('Standardized Residuals')
ax.set_xlabel('Theoretical Quantiles')
ax.plot([np.min([x,y]),np.max([x,y])],[np.min([x,y]),np.max([x,y])], color = 'r', ls = '--')
for val in top3.index:
    ax.annotate(val,xy=(df_OLS['theoretical_quantiles'].loc[val],df_OLS['sorted_student_residuals'].loc[val]))
plt.show()









# 3rd diagnostic plot: Scale-location plot
student_residuals_OLS = results.get_influence().resid_studentized_internal
sqrt_student_residuals_OLS = pd.Series(np.sqrt(np.abs(student_residuals_OLS)))
sqrt_student_residuals_OLS.index = results.resid.index
smoothed = lowess(sqrt_student_residuals_OLS,fitted_OLS)
top3 = abs(sqrt_student_residuals_OLS).sort_values(ascending = False)[:3]

fig, ax = plt.subplots()
ax.scatter(fitted_OLS, sqrt_student_residuals_OLS, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
ax.set_xlabel('Fitted Values')
ax.set_title('Scale-Location')
ax.set_ylim(0,max(sqrt_student_residuals_OLS)+0.1)
for i in top3.index:
    ax.annotate(i,xy=(fitted_OLS[i],sqrt_student_residuals_OLS[i]))
plt.show()








# 4th diagnostic plot: residuals vs leverage plot
student_residuals_OLS = pd.Series(results.get_influence().resid_studentized_internal)
student_residuals_OLS.index = results.resid.index
df_OLS = pd.DataFrame(student_residuals_OLS)
df_OLS.columns = ['student_residuals']
df_OLS['leverage'] = results.get_influence().hat_matrix_diag
smoothed = lowess(df_OLS['student_residuals'],df_OLS['leverage'])
sorted_student_residuals = abs(df_OLS['student_residuals']).sort_values(ascending = False)
top3 = sorted_student_residuals[:3]

fig, ax = plt.subplots()
x = df_OLS['leverage']
y = df_OLS['student_residuals']
xpos = max(x)+max(x)*0.01  
ax.scatter(x, y, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('Studentized Residuals')
ax.set_xlabel('Leverage')
ax.set_title('Residuals vs. Leverage')
ax.set_ylim(min(y)-min(y)*0.15,max(y)+max(y)*0.15)
ax.set_xlim(-0.01,max(x)+max(x)*0.05)
plt.tight_layout()
for val in top3.index:
    ax.annotate(val,xy=(x.loc[val],y.loc[val]))

cooksx = np.linspace(min(x), xpos, 50)
p = len(results.params)
poscooks1y = np.sqrt((p*(1-cooksx))/cooksx)
poscooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx)
negcooks1y = -np.sqrt((p*(1-cooksx))/cooksx)
negcooks05y = -np.sqrt(0.5*(p*(1-cooksx))/cooksx)

ax.plot(cooksx,poscooks1y,label = "Cook's Distance", ls = ':', color = 'r')
ax.plot(cooksx,poscooks05y, ls = ':', color = 'r')
ax.plot(cooksx,negcooks1y, ls = ':', color = 'r')
ax.plot(cooksx,negcooks05y, ls = ':', color = 'r')
ax.plot([0,0],ax.get_ylim(), ls=":", alpha = .3, color = 'k')
ax.plot(ax.get_xlim(), [0,0], ls=":", alpha = .3, color = 'k')
ax.annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
ax.annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
ax.annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
ax.annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')
ax.legend()
plt.show()







# Fit measuremenmts for the linear regression
y_predicted_OLS=results.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predicted_OLS))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predicted_OLS))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predicted_OLS)))
predR2_OLS=metrics.r2_score(y_test, y_predicted_OLS)






# CART (Regression trees)
classifier=tree.DecisionTreeRegressor()
clf=classifier.fit(X_train,y_train)
predicted_values_CART=clf.predict(X=X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted_values_CART))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predicted_values_CART))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted_values_CART)))
predR2_CART=metrics.r2_score(y_test, predicted_values_CART)









# Cross validation
#from sklearn.model_selection import KFold
#kf = KFold(n_splits=2) # Define the split - into 2 folds 
#kf.get_n_splits(data2) #  returns the number of splitting iterations in the cross-validator
#print(kf)
#from sklearn.cross_validation import cross_val_score, cross_val_predict

# Perform 6-fold cross validation
#lm = linear_model.LinearRegression()
#mm=lm.fit(X_train, y_train)
#predictions_CV6 = cross_val_predict(mm, X_test, y_test, cv=6)
#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions_CV6))  
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions_CV6))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions_CV6)))
#accuracy_CV6 = metrics.r2_score(y_test, predictions_CV6)
#print ("Cross-Predicted Accuracy:", accuracy_CV6)










# Random forest
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, oob_score=True)
# Train the model on training data
forest=rf.fit(X_train, y_train);
# Use the forest's predict method on the test data
rf_predictions = rf.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rf_predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, rf_predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rf_predictions)))
predR2_RF=metrics.r2_score(y_test, rf_predictions)







# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(data2.columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];







# New random forest with only the four most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42, oob_score=True)
# Extract the four most important features
features=data2.columns
important_indices = [which(features=='displacement'), which(features=='weight'), which(features=='horsepower'),which(features=='acceleration')]
#train_important = X_train[:, important_indices]
#test_important = X_test[:, important_indices]
important_indices=[1,2,3,4]
train_important = X_train.filter(items=['displacement','weight','horsepower','acceleration'])
test_important = X_test.filter(items=['displacement','weight','horsepower','acceleration'])
# Train the random forest
forest_important=rf_most_important.fit(train_important, y_train)
# Make predictions and determine the error
rf_important_predictions = rf_most_important.predict(test_important)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rf_important_predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, rf_important_predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rf_important_predictions)))






# Ridge Regression
mm=Ridge()
parameters={'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 2, 5, 10]}
ridge_regressor=GridSearchCV(mm, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X_train, y_train)
print(ridge_regressor.best_params_) # Best alpha
print(ridge_regressor.best_score_) 
rr=Ridge(alpha=ridge_regressor.best_params_['alpha'])
result_ridge=rr.fit(X_train, y_train)
print(result_ridge.coef_)
ridge_predict=result_ridge.predict(X_test)
# fit measurements with ridge regression
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, ridge_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, ridge_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, ridge_predict)))
predR2_ridge=metrics.r2_score(y_test, ridge_predict)






# Lasso Regression
lasso=Lasso()
parameters={'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 2, 5, 10]}
lasso_regressor=GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X_train, y_train)
print(lasso_regressor.best_params_) # Best alpha
print(lasso_regressor.best_score_) # Lowest possible MSE with Lasso Regression and the best alpha
ll=Lasso(alpha=lasso_regressor.best_params_['alpha'])
result_lasso=ll.fit(X_train, y_train)
lasso_predict=result_lasso.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, lasso_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, lasso_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, lasso_predict)))
predR2_lasso=metrics.r2_score(y_test, lasso_predict)
















