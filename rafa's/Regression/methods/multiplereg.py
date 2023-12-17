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
X_orgi = frame
X_test_orgi = df_test
y = frame[['duration_ms', 'popularity', 'danceability']]
Y_test = df_test[['duration_ms', 'popularity', 'danceability']]


#SIMPLE MULTIPLE REGRESSION FOR ALL THE CONTINUOUS ATTRIBUTES AGAINST THE THREE SELECTED TARGETS

def multireg(X,y,X_test,Y_test):
    results = {}
    targets = y.columns
    for target in list(targets):
        X = X_orgi[X_orgi.columns.difference([target])]
        X_test = X_test_orgi[X_test_orgi.columns.difference([target])]
        x_train = X.values
        print(X.columns)
        y_train = y[target].values

        x_test = X_test.values
        y_test = Y_test[target].values

        reg = LinearRegression()
        reg.fit(x_train, y_train)

        y_pred = reg.predict(x_test)

        results[f'Coefficients & Intercept & R2 & MSE & MAE: {target}'] = (reg.coef_,reg.intercept_,r2_score(y_test, y_pred),mean_squared_error(y_test, y_pred),mean_absolute_error(y_test, y_pred))

        print(f'Coefficients & Intercept & R2 & MSE & MAE: {target}'+'\n', reg.coef_ , '\n', reg.intercept_ )
        print('R2: %.3f' % r2_score(y_test, y_pred))
        print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
        print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

        for column_names in X.columns:
            sns.scatterplot(data=df_test, x=column_names, y=target, label="True")
            sns.scatterplot(data=df_test, x=column_names, y=reg.predict(X_test), label="Predicted", marker="X")
            plt.suptitle("Scatter "+str(target)+" "+str(column_names))
            plt.legend()
            #plt.show()

    return results

results_reg = multireg(X_orgi,y,X_test_orgi,Y_test)
print(results_reg)
 


 #MULTIPLE REGRESSION BUT WITH RIDGE REGULARIZATION

def multiRIDGEreg(X,y,X_test,Y_test):
    results = {}
    targets = y.columns
    for target in list(targets):
        X = X_orgi[X_orgi.columns.difference([target])]
        X_test = X_test_orgi[X_test_orgi.columns.difference([target])]
        x_train = X.values
        print(X.columns)
        y_train = y[target].values

        x_test = X_test.values
        y_test = Y_test[target].values

        reg = Ridge()
        reg.fit(x_train, y_train)

        y_pred = reg.predict(x_test)

        results[f'RIDGE Coefficients & Intercept & R2 & MSE & MAE: {target}'] = (reg.coef_,reg.intercept_,r2_score(y_test, y_pred),mean_squared_error(y_test, y_pred),mean_absolute_error(y_test, y_pred))

        print(f'RIDGE Coefficients & Intercept & R2 & MSE & MAE: {target}'+'\n', reg.coef_ , '\n', reg.intercept_ )
        print('R2: %.3f' % r2_score(y_test, y_pred))
        print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
        print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

        for column_names in X.columns:
            sns.scatterplot(data=df_test, x=column_names, y=target, label="True")
            sns.scatterplot(data=df_test, x=column_names, y=reg.predict(X_test), label="Predicted", marker="X")
            plt.suptitle("Scatter "+str(target)+" "+str(column_names)+" RIDGE")
            plt.legend()
            #plt.show()

    return results


results_RIDGE_reg = multiRIDGEreg(X_orgi,y,X_test_orgi,Y_test)
print(results_RIDGE_reg)



#MULTIPLE REGRESSION BUT WITH LASSO REGULARIZATION


def multiLASSOreg(X,y,X_test,Y_test):
    results = {}
    targets = y.columns
    for target in list(targets):
        X = X_orgi[X_orgi.columns.difference([target])]
        X_test = X_test_orgi[X_test_orgi.columns.difference([target])]
        x_train = X.values
        print(X.columns)
        y_train = y[target].values

        x_test = X_test.values
        y_test = Y_test[target].values

        reg = Lasso()
        reg.fit(x_train, y_train)

        y_pred = reg.predict(x_test)

        results[f'LASSO Coefficients & Intercept & R2 & MSE & MAE: {target}'] = (reg.coef_,reg.intercept_,r2_score(y_test, y_pred),mean_squared_error(y_test, y_pred),mean_absolute_error(y_test, y_pred))

        print(f'LASSO Coefficients & Intercept & R2 & MSE & MAE: {target}'+'\n', reg.coef_ , '\n', reg.intercept_ )
        print('R2: %.3f' % r2_score(y_test, y_pred))
        print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
        print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

        for column_names in X.columns:
            sns.scatterplot(data=df_test, x=column_names, y=target, label="True")
            sns.scatterplot(data=df_test, x=column_names, y=reg.predict(X_test), label="Predicted", marker="X")
            plt.suptitle("Scatter "+str(target)+" "+str(column_names)+" LASSO")
            plt.legend()
            #plt.show()

    return results


results_LASSO_reg = multiLASSOreg(X_orgi,y,X_test_orgi,Y_test)
print(results_LASSO_reg)