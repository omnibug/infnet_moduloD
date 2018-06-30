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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
#from sklearn.tree import export_graphviz
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import datetime as dt

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
        if len(df[col].unique()) < 1:
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
#                df = df.drop([col], axis=1)
    return df

def plot_graph(importances, indices, predictor_var, lmodel):
#    print '\nAttribute importance order for '+lmodel
    xvar = []
    for i in indices:
        #        print '{0:2d} - {1:s}'.format(i, predictor_var[i])
        xvar.append(predictor_var[i])
    # Plot the feature importances
    plt.figure(figsize=(26, 6), dpi=200)
    plt.title(u'Importância dos Atributos para '+lmodel)
    plt.grid(b=True, which='both', color='0.7',linestyle='-', axis='y')
    plt.xticks(range(len(indices)), xvar, rotation=90 )
    plt.bar(range(len(indices)), importances[indices], color="#4169E1", align="center",edgecolor="#4169E1")
    plt.xlabel(u"Atributos")
    plt.ylabel(u"Importância")
    plt.savefig(test_nbr+'_'+lmodel+'.png', dpi=200)
    plt.show()
    
print "\nRead CSV"
#Reading the dataset in a dataframe using Pandas
df_full = pd.read_csv('myosotis_database_encoded.csv') 

"""
                'peso_round',
                'altura_round',
                'peso_aproximado', 
                'altura_aproximada', 
                'idade', 
                'dias_desaparecido', 
                'anos_desaparecido', 
                'isexo', 
                'isexo_0', 'isexo_1', 
                'iolhos_0', 'iolhos_1', 'iolhos_2', 'iolhos_3', 'iolhos_4', 
                'icor_da_pele_0','icor_da_pele_1','icor_da_pele_2','icor_da_pele_3','icor_da_pele_4','icor_da_pele_5','icor_da_pele_6', 
                'icabelo', 
                'icabelo_0', 'icabelo_1', 'icabelo_2', 'icabelo_3', 'icabelo_4', 'icabelo_5', 'icabelo_6', 
                'itipo_fisico_0', 'itipo_fisico_1', 'itipo_fisico_2', 'itipo_fisico_3', 'itipo_fisico_4',
                'dias_round',
                'idade_round'
                'idade_round_5','idade_round_6','idade_round_7','idade_round_8','idade_round_9','idade_round_10','idade_round_11','idade_round_12','idade_round_13','idade_round_14','idade_round_15','idade_round_16',
                
                'peso_round',
                'peso_round_0','peso_round_1','peso_round_2','peso_round_3','peso_round_4','peso_round_5','peso_round_6','peso_round_7','peso_round_8','peso_round_9','peso_round_10','peso_round_11','peso_round_12','peso_round_13','peso_round_14','peso_round_15','peso_round_16','peso_round_17','peso_round_18',
"""
#cor, sexo, olhos, idade e dias desaparecido.
test_nbr = 'teste2_01'
predictor_var = [
                'dias_round',
                'altura_round',
                'idade_round', 
                'peso_round',
                'itipo_fisico',
                'icor_da_pele', 
                'isexo', 
                'icabelo', 
                'itranstorno_mental',
                'iolhos'
                ]

s_dt = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')

with open('encoded_data.json', 'r') as f:
    rdata = f.read()
    l_encoded_data = json.loads(rdata)
f.closed

outcome_var = 'istatus'
lst_remove = ['id','nome','data_nascimento','data_desaparecimento']
df_full = df_full.drop(lst_remove, axis=1)

df_full = one_hot_encoder(df_full)
print df_full.info()
                
# Copiando os dados do csv
data = df_full[[outcome_var]+predictor_var].values.copy()
columns = df_full[[outcome_var]+predictor_var].columns.copy()

# Embaralhando os dados para garantir aleatoriedade entre as amostras
np.random.shuffle(data)

# Separando atributos de classes
x = data[:, 1:]

y = list(data[:, 0])

# 70% dos dados serao utilizados para treinamento e 30% para o teste
# A divisao sera estratificada, serao mantidas as proporcoes de spam e nao spam em cada grupo

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, train_size=0.8, stratify=y)

print '\n Prediction Variables :'+ str(predictor_var)
print '\n Output Variable :'+ outcome_var


#DECISION TREE        
lmodel = 'DecisionTreeClassifier'
print '\n'+lmodel

var_model = {'splitter': 'random', 'max_leaf_nodes': 10, 'min_samples_leaf': 14, 'min_samples_split': 50, 'random_state': 3, 'criterion': 'entropy', 'max_features': None, 'max_depth': 3}
model = DecisionTreeClassifier(**var_model)
print model
model.fit(x_treino,y_treino)
#Make predictions on training set: 
predictions = model.predict(x_teste)
#Print accuracy 
accuracy = metrics.accuracy_score(predictions,y_teste) 
print "Accuracy : %s" % "{0:.4%}".format(accuracy)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plot_graph(importances, indices, predictor_var, lmodel)


#RANDOM FOREST
lmodel = 'RandomForestClassifier'
print '\n'+lmodel
var_model = {'n_jobs': 1, 'max_leaf_nodes': 30, 'min_samples_leaf': 10, 'n_estimators': 60, 'max_features': 'log2', 'random_state': None, 'criterion': 'gini', 'min_samples_split': 5, 'max_depth': 8}
model = RandomForestClassifier(**var_model)
print model
model.fit(x_treino,y_treino)

#Make predictions on training set: 
predictions = model.predict(x_teste)

#Print accuracy 
accuracy = metrics.accuracy_score(predictions,y_teste) 

print "Accuracy : %s" % "{0:.4%}".format(accuracy)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plot_graph(importances, indices, predictor_var, lmodel)

#LOGISTIC REGRESSION
lmodel = 'LogisticRegression'
print '\n'+lmodel
var_model = {'warm_start': False, 'C': 1.0, 'n_jobs': 1, 'verbose': 0, 'intercept_scaling': 1, 'fit_intercept': True, 'max_iter': 300, 'penalty': 'l2', 'multi_class': 'ovr', 'random_state': None, 'dual': False, 'tol': 0.0001, 'solver': 'newton-cg', 'class_weight': None}
model = LogisticRegression(**var_model)
print model
model.fit(x_treino,y_treino)
#Make predictions on training set: 
predictions = model.predict(x_teste)
#Print accuracy 
accuracy = metrics.accuracy_score(predictions,y_teste) 
print "Accuracy : %s" % "{0:.4%}".format(accuracy)

importances = model.coef_ 
indices = np.argsort(importances)[::-1]
indices = indices[0]
plot_graph(importances[0], indices, predictor_var, lmodel)

#XGB
lmodel = 'XGBClassifier'
# 'learning_rate':0.1, 'n_estimators':6000, 'max_depth':1, 'min_child_weight':1,  'gamma':1.1,  'subsample':0.8,  'colsample_bytree':0.9,  'reg_alpha':0.0, 'objective':'binary:logistic', 'nthread':4, 'scale_pos_weight':1, 'seed':27
var_model = {
'learning_rate':0.001, 'n_estimators':5000, 
'max_depth':4, 'min_child_weight':1,  
'gamma':0.0,  
'subsample':0.1,  'colsample_bytree':0.8,  
'objective':'reg:logistic', 
'missing':np.nan, 'silent':0, 'seed':7 }
model = XGBClassifier(**var_model)
print '\n'+lmodel
print model
model.fit(x_treino,y_treino)
#Make predictions on training set: 
predictions = model.predict(x_teste)
#Print accuracy 
accuracy = metrics.accuracy_score(predictions,y_teste) 
print "Accuracy : %s" % "{0:.4%}".format(accuracy)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

import codecs  
f=codecs.open('xgb_tree6.png', mode='wb')  
g=xgb.to_graphviz(model, num_trees=6, rankdir='LR')  
f.write(g.pipe('png'));  
f.close()  

#plt.figure(figsize=(26, 6), dpi=400)
#plt.savefig(test_nbr+'_tree_'+lmodel+'.png', dpi=200)
#plt.show()

print '\nFinished'