# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 16:05:01 2016

@author: Carlos
https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
#from sklearn.tree import export_graphviz
from sklearn import metrics
import xgboost as xgb

print '='*200
print '\nStarted'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Definindo uma funcao de listagem e descricao do dataframe
def desc_df(df,n):
    print '\nData File data and stats'
    print df.head(n)
    print df.describe()

# Update NULL with the most common value in column
def imputer_most_common(r):
    r.fillna(r.value_counts().idxmax(),inplace=True)

def common_values(df):
    var_columns = df.columns
    for col in var_columns:
        if (col <> outcome_var):
            if len(df[col].unique()) < 90:
                # updates null columns with the most common value in the column
                imputer_most_common(df[col])
                print 'Different values = '+str(len(df[col].unique()))+'\t for '+col
            else:
                # updates null columns with the most common value in the column
                imputer_most_common(df[col])
                print 'Too many values = '+str(len(df[col].unique()))+'\t for '+col
                if df[col].isnull().values.any():
                   print 'Null values found in '+col
    return df

def one_hot_encoder(df):
    enc=OneHotEncoder(sparse=False)
    le = LabelEncoder()
    var_columns = df.columns
    for col in var_columns:
        if len(df[col].unique()) < 10:
            df[col] = le.fit_transform(df[col])
            if col <> outcome_var:
                # creating an exhaustive list of all possible categorical values
                data=df[[col]]
                enc.fit(data)
                # Fitting One Hot Encoding on train data
                temp = enc.transform(df[[col]])
                # Changing the encoded features into a data frame with new column names
                temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
                # In side by side concatenation index values should be same
                # Setting the index values similar to the X_train data frame
                temp=temp.set_index(data.index.values)
                # adding the new One Hot Encoded varibales to the train data frame
                df=pd.concat([df,temp],axis=1)
                df = df.drop([col], axis=1)
    return df

print "\nRead CSV"
#Reading the dataset in a dataframe using Pandas
df_full = pd.read_csv('myosotis_database_encoded.csv') 

with open('encoded_data.json', 'r') as f:
    rdata = f.read()
    l_encoded_data = json.loads(rdata)
f.closed

print "\df_full File"
desc_df(df_full,5)

outcome_var = 'istatus'
lst_remove = ['id','nome','data_nascimento','data_desaparecimento']
df_full = df_full.drop(lst_remove, axis=1)

#df_full = common_values(df_full)

df_full = one_hot_encoder(df_full)
                
print "\df_full File"
desc_df(df_full,5)

# Copiando os dados do csv
data = df_full.values.copy()
columns = df_full.columns.copy()
print columns

# Embaralhando os dados para garantir aleatoriedade entre as amostras
np.random.shuffle(data)

# Separando atributos de classes
x = data[:, 1:]

#x = x/np.amax(x, axis=0)

y = list(data[:, 0])
print(data[:5])

# 70% dos dados serao utilizados para treinamento e 30% para o teste
# A divisao sera estratificada, serao mantidas as proporcoes de spam e nao spam em cada grupo

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, train_size=0.7, stratify=y)

print(x_treino[:5])
print(x_teste[:5])
print(y_treino[:5])
print(y_teste[:5])


#Generic function for making a classification model and accessing performance: 
def classification_model(model, data, predictors, outcome):
    #Fit the model: 
    model.fit(data[predictors],data[outcome])
    #Make predictions on training set: 
    predictions = model.predict(data[predictors])
    #Print accuracy 
    accuracy = metrics.accuracy_score(predictions,data[outcome]) 
    print "Accuracy : %s" % "{0:.4%}".format(accuracy)
    #Perform k-fold cross-validation with 5 folds 
    kf = KFold(data.shape[0], n_folds=5) 
    error = [] 
    for train, test in kf:
        # Filter training data 
        train_predictors = (data[predictors].iloc[train,:])
        # The target we're using to train the algorithm. 
        train_target = data[outcome].iloc[train]
        # Training the algorithm using the predictors and target.    
        model.fit(train_predictors, train_target)
        #Record error from each cross-validation run    
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    print "Cross-Validation Score : %s" % "{0:.4%}".format(np.mean(error))
    #Fit the model again so that it can be refered outside the function: 
    model.fit(data[predictors],data[outcome])
    return np.mean(error)

#Generic function for making a classification model and accessing performance: 
def classify_model(model, data, predictors, outcome):
    #Fit the model: 
    model.fit(data[predictors],data[outcome])
    #Perform k-fold cross-validation with 5 folds 
    kf = KFold(data.shape[0], n_folds=5) 
    error = [] 
    for train, test in kf:
        # Filter training data 
        train_predictors = (data[predictors].iloc[train,:])
        # The target we're using to train the algorithm. 
        train_target = data[outcome].iloc[train]
        # Training the algorithm using the predictors and target.    
        model.fit(train_predictors, train_target)
        #Record error from each cross-validation run    
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    #Fit the model again so that it can be refered outside the function: 
    model.fit(data[predictors],data[outcome])
    return np.mean(error)

parametersDecisionTree=[
                        ['splitter',['best','random']],
                        ['max_leaf_nodes',[10, 8, 6, 4, 2]],
                        ['min_samples_leaf',[11, 9, 7, 5, 3, 1]],
                        ['max_features',['auto','log2',None]],
                        ['random_state',[3,2,1]],
                        ['criterion',['gini','entropy']],
                        ['min_samples_split',[10, 90, 60, 30, 2]],
                        ['max_depth',[20, 15, 10, 5, 3, 1]]
                        ]

parametersDecisionTreeTunning=[
                        ['splitter',['random']],
                        ['max_leaf_nodes',[10]],
                        ['min_samples_leaf',[14]],
                        ['max_features',[None]],
                        ['random_state',[3]],
                        ['criterion',['entropy']],
                        ['min_samples_split',[50]],
                        ['max_depth',[3]]
                        ]

parametersRandomForests=[
                        ['n_jobs',[1]],
                        ['max_leaf_nodes',[20, 10]],
                        ['min_samples_leaf',[20, 10]],
                        ['n_estimators',[200,100,50,25,10,5,3,2]],
                        ['min_samples_split',[50,40,30,2]],
                        ['random_state',[None,1,2]],
                        ['criterion',['gini','entropy']],
                        ['max_features',['auto','log2',None]],
                        ['max_depth',[20, 15, 5, 1, None]]
                        ]

parametersRandomForestsTuning=[
                        ['n_jobs',[1]],
                        ['max_leaf_nodes',[30]],
                        ['min_samples_leaf',[10]],
                        ['n_estimators',[60]],
                        ['min_samples_split',[5]],
                        ['random_state',[None]],
                        ['criterion',['gini']],
                        ['max_features',['log2']],
                        ['max_depth',[8]]
                        ]

parametersLogisticRegression=[
                        ['C',[0.1, 0.5, 1.0]],
                        ['class_weight',[None]],
                        ['dual',[False]],
                        ['fit_intercept',[True]],
                        ['intercept_scaling',[1]],
                        ['max_iter',[300,100]],
                        ['multi_class',['ovr',]],
                        ['n_jobs',[1]],
                        ['penalty',['l2']],
                        ['random_state',[None]],
                        ['solver',['newton-cg', 'lbfgs', 'liblinear', 'sag']],
                        ['tol',[0.0001]],
                        ['verbose',[0]],
                        ['warm_start',[False]]
                        ]

parametersLogisticRegressionTuning=[
                        ['C',[1.0]],
                        ['class_weight',[None]],
                        ['dual',[False]],
                        ['fit_intercept',[True]],
                        ['intercept_scaling',[1]],
                        ['max_iter',[300]],
                        ['multi_class',['ovr',]],
                        ['n_jobs',[1]],
                        ['penalty',['l2']],
                        ['random_state',[None]],
                        ['solver',['newton-cg']],
                        ['tol',[0.0001]],
                        ['verbose',[0]],
                        ['warm_start',[False]]
                        ]

dicmodel = {}
maxscore=0  
maxdicmodel = {}
def run_model(i, parameters, lmodel,df, predictor_var, outcome_var):
    global maxscore
    global maxdicmodel
    if i < len(parameters):
        for j in range(len(parameters[i][1])):
            dicmodel[parameters[i][0]] = parameters[i][1][j]
            run_model(i+1, parameters, lmodel,df, predictor_var, outcome_var)
    else:
        if lmodel == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(**dicmodel)
        elif lmodel == 'RandomForestClassifier':
            model = RandomForestClassifier(**dicmodel)
        elif lmodel == 'LogisticRegression':
            model = LogisticRegression(**dicmodel)
        elif lmodel == 'XGBClassifier':
            model = xgb.XGBClassifier(**dicmodel)
        else:
            print 'ERROR'
        temp = classify_model(model, df,predictor_var,outcome_var)
        if lmodel == 'LogisticRegression':
            print dicmodel,round(temp*100,4),round(maxscore*100,4),"                         \r"
        if temp >= maxscore:
            if temp > maxscore:
                print "Cross-Validation Score : %s" % "{0:.4%}".format(temp)
                print dicmodel
            maxscore=temp
            maxdicmodel=dicmodel.copy()

def prepare_model(lmodel, df, parameters):
    print '\nTry and tune various models for ' + lmodel
    print 'Model: '+lmodel+' for: '+str(predictor_var)
    run_model(0, parameters, lmodel,df, predictor_var, outcome_var)

def submit_file(lmodel, df, test, parameters):
    print '\nFit and submit file for ' + lmodel
    predictor_var = df.columns.values.tolist()[2:]
#    predictor_var.remove(outcome_var)
    print 'Model: '+lmodel+' for: '+str(predictor_var)
    print parameters
    if lmodel == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(**parameters)
    elif lmodel == 'RandomForestClassifier':
        model = RandomForestClassifier(**parameters)
    elif lmodel == 'LogisticRegression':
        model = LogisticRegression(**parameters)
    elif lmodel == 'XGBClassifier':
        model = xgb.XGBClassifier(**parameters)
    else:
        print 'ERROR IN lmodel='+lmodel
    model.fit(df[predictor_var],df[outcome_var])
    prediction_lst = model.predict(test[predictor_var])
    # Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
    df_submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction_lst })
    # Write the solution to file
    solutionfile = lmodel+'_Solution.csv'
    df_submission.to_csv(solutionfile, index=False)
    print 'Wrote file ' + solutionfile
    

print df_full.columns.values.tolist()[1:]
predictor_var = df_full.columns.values.tolist()[1:]


#DECISION TREE        
dicmodel = {}
maxscore=0  
maxdicmodel = {}
lmodel = 'DecisionTreeClassifier'
prepare_model(lmodel, df_full, parametersDecisionTreeTunning)
var_model = maxdicmodel.copy()
model = DecisionTreeClassifier(**var_model)
print model
print "7/3"
#model.fit(df_full[predictor_var],df_full[outcome_var])
model.fit(x_treino,y_treino)
#Make predictions on training set: 
predictions = model.predict(x_teste)
#Print accuracy 
accuracy = metrics.accuracy_score(predictions,y_teste) 
print "Accuracy : %s" % "{0:.4%}".format(accuracy)


importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
imp = ''
for i in indices:
    #print str(i)+' - '+predictor_var[i]
    imp = imp+str(i)+', '
print imp

# Plot the feature importances
plt.figure(figsize=(26, 6), dpi=200)
plt.title(u"Importância dos Atributos")
plt.grid(b=True, which='both', color='0.7',linestyle='-', axis='y')
plt.xticks(range(len(indices)), indices, rotation=90 )
plt.bar(range(len(indices)), importances[indices], color="#4169E1", align="center",edgecolor="#4169E1")
plt.xlabel(u"Atributos")
plt.ylabel(u"Importância")
plt.show()

#submit_file(lmodel, train, test, var_model)
"""
#RANDOM FOREST
dicmodel = {}
maxscore=0  
maxdicmodel = {}
lmodel = 'RandomForestClassifier'
prepare_model(lmodel, train, parametersRandomForestsTuning)
var_model = maxdicmodel.copy()
model = RandomForestClassifier(**var_model)
print model
predictor_var = train.columns.values.tolist()[2:]
model.fit(train[predictor_var],train[outcome_var])
#Make predictions on training set: 
predictions = model.predict(train[predictor_var])
#Print accuracy 
accuracy = metrics.accuracy_score(predictions,train[outcome_var]) 
print "Accuracy : %s" % "{0:.4%}".format(accuracy)
#submit_file(lmodel, train, test, var_model)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for i in indices:
    print str(i)+' - '+predictor_var[i]
# Plot the feature importances
plt.figure(figsize=(26, 6), dpi=200)
plt.title(u"Importância dos Atributos")
plt.grid(b=True, which='both', color='0.7',linestyle='-', axis='y')
plt.xticks(range(len(indices)), indices, rotation=90 )
plt.bar(range(len(indices)), importances[indices], color="#4169E1", align="center",edgecolor="#4169E1")
plt.xlabel(u"Atributos")
plt.ylabel(u"Importância")
plt.show()

#LOGISTIC REGRESSION
dicmodel = {}
maxscore=0  
maxdicmodel = {}
lmodel = 'LogisticRegression'
prepare_model(lmodel, train, parametersLogisticRegressionTuning)
var_model = maxdicmodel.copy()
model = LogisticRegression(**var_model)
print model
predictor_var = train.columns.values.tolist()[2:]
model.fit(train[predictor_var],train[outcome_var])
#Make predictions on training set: 
predictions = model.predict(train[predictor_var])
#Print accuracy 
accuracy = metrics.accuracy_score(predictions,train[outcome_var]) 
print "Accuracy : %s" % "{0:.4%}".format(accuracy)
#submit_file(lmodel, train, test, var_model)
"""
print '\nFinished'