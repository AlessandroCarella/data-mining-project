# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import os



#Best targets : duration, popularity, danceability.

#set wd
os.chdir("methods")

#load ds normalized
df = pd.read_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)
df.drop(columns='genre', inplace=True)

df_train, df_test = train_test_split(df, test_size=0.3, random_state=100)
#Set y,x
frame = df_train
X = frame
X_test = df_test
y = frame[['duration_ms', 'popularity', 'danceability']]
Y_test = df_test[['duration_ms', 'popularity', 'danceability']]


#SIMPLE REGRESSION FOR ALL THE CONTINUOUS ATTRIBUTES AGAINST THE THREE SELECTED TARGETS

def simplereg(X,y,X_test,Y_test):
    results = {}
    targets = y.columns
    for target in list(targets):
        for column_names in df.columns:
            if target == column_names:
                continue
            x_train = X[column_names].values.reshape(-1, 1)
            y_train = y[target].values

            x_test = X_test[column_names].values.reshape(-1, 1)
            y_test = Y_test[target].values

            reg = LinearRegression()
            reg.fit(x_train, y_train)



            y_pred = reg.predict(x_test)

            results[f'Coefficients & Intercept & R2 & MSE & MAE: {target}, {column_names}'] = (reg.coef_,reg.intercept_,r2_score(y_test, y_pred),mean_squared_error(y_test, y_pred),mean_absolute_error(y_test, y_pred))

            print(f'Coefficients & Intercept & R2 & MSE & MAE: {target}, {column_names}'+'\n', reg.coef_ , '\n', reg.intercept_ )
            print('R2: %.3f' % r2_score(y_test, y_pred))
            print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
            print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

            sns.scatterplot(data=df_train, x=column_names, y=target)
            plt.plot(x_train, reg.coef_[0]*x_train+reg.intercept_, c="red")
            #plt.show()

    return results


results_reg = simplereg(X,y,X_test,Y_test)
results_reg



#SIMPLE REGRESSION BUT WITH RIDGE REGULARIZATION

def simpleRIDGEreg(X,y,X_test,Y_test):
    results = {}
    targets = y.columns
    for target in list(targets):
        for column_names in df.columns:
            if target == column_names:
                continue
            x_train = X[column_names].values.reshape(-1, 1)
            y_train = y[target].values

            x_test = X_test[column_names].values.reshape(-1, 1)
            y_test = Y_test[target].values

            reg = Ridge()
            reg.fit(x_train, y_train)



            y_pred = reg.predict(x_test)

            results[f'RIDGE-Coefficients & Intercept & R2 & MSE & MAE: {target}, {column_names}'] = (reg.coef_,reg.intercept_,r2_score(y_test, y_pred),mean_squared_error(y_test, y_pred),mean_absolute_error(y_test, y_pred))

            print(f'RIDGE-Coefficients & Intercept & R2 & MSE & MAE: {target}, {column_names}'+'\n', reg.coef_ , '\n', reg.intercept_ )
            print('R2: %.3f' % r2_score(y_test, y_pred))
            print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
            print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

            sns.scatterplot(data=df_train, x=column_names, y=target)
            plt.plot(x_train, reg.coef_[0]*x_train+reg.intercept_, c="red")
            #plt.show()

    return results


results_RIDGE_reg = simpleRIDGEreg(X,y,X_test,Y_test)
results_RIDGE_reg



#SIMPLE REGRESSION BUT WITH LASSO REGULARIZATION

def simpleLASSOreg(X,y,X_test,Y_test):
    results = {}
    targets = y.columns
    for target in list(targets):
        for column_names in df.columns:
            if target == column_names:
                continue
            x_train = X[column_names].values.reshape(-1, 1)
            y_train = y[target].values

            x_test = X_test[column_names].values.reshape(-1, 1)
            y_test = Y_test[target].values

            reg = Lasso()
            reg.fit(x_train, y_train)



            y_pred = reg.predict(x_test)

            results[f'LASSO-Coefficients & Intercept & R2 & MSE & MAE: {target}, {column_names}'] = (reg.coef_,reg.intercept_,r2_score(y_test, y_pred),mean_squared_error(y_test, y_pred),mean_absolute_error(y_test, y_pred))

            print(f'LASSO-Coefficients & Intercept & R2 & MSE & MAE: {target}, {column_names}'+'\n', reg.coef_ , '\n', reg.intercept_ )
            print('R2: %.3f' % r2_score(y_test, y_pred))
            print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
            print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

            sns.scatterplot(data=df_train, x=column_names, y=target)
            plt.plot(x_train, reg.coef_[0]*x_train+reg.intercept_, c="red")
            #plt.show()

    return results


results_LASSO_reg = simpleLASSOreg(X,y,X_test,Y_test)
results_LASSO_reg