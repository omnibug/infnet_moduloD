# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:56:02 2016

@author: Carlos
"""

import pandas as pd
from sklearn import metrics

#Definindo uma funcao de listagem e descricao do dataframe
def desc_df(df,n):
    print '\nData File data and stats'
    print df.head(n)
    print df.describe()

print '='*200
print '\nStarted'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

outcome_var = 'istatus'
predicted_var = 'Prediction'

pred = pd.read_csv("myosotis_database_prediction_1.csv") 
accuracy1 = metrics.accuracy_score(pred[predicted_var],pred[outcome_var]) 
print "Accuracy : %s" % "{0:.4%}".format(accuracy1)

pred = pd.read_csv("myosotis_database_prediction_1.csv") 
accuracy = metrics.accuracy_score(pred[predicted_var],pred[outcome_var]) 
print "\nAccuracy : %s" % "{0:.4%}".format(accuracy)
pred1 = pred[ pred[outcome_var] == 1 ]
pred0 = pred[ pred[outcome_var] == 0 ]
accuracy = metrics.accuracy_score(pred1[predicted_var],pred1[outcome_var]) 
print "Accuracy 1 : %s" % "{0:.4%}".format(accuracy)
accuracy = metrics.accuracy_score(pred0[predicted_var],pred0[outcome_var]) 
print "Accuracy 0 : %s" % "{0:.4%}".format(accuracy)

pred = pd.read_csv("myosotis_database_prediction_linux_1.csv") 
accuracy = metrics.accuracy_score(pred[predicted_var],pred[outcome_var]) 
print "\nAccuracy : %s" % "{0:.4%}".format(accuracy)
pred1 = pred[ pred[outcome_var] == 1 ]
pred0 = pred[ pred[outcome_var] == 0 ]
accuracy = metrics.accuracy_score(pred1[predicted_var],pred1[outcome_var]) 
print "Accuracy 1 : %s" % "{0:.4%}".format(accuracy)
accuracy = metrics.accuracy_score(pred0[predicted_var],pred0[outcome_var]) 
print "Accuracy 0 : %s" % "{0:.4%}".format(accuracy)

pred = pd.read_csv("myosotis_database_prediction_linux_2.csv") 
accuracy = metrics.accuracy_score(pred[predicted_var],pred[outcome_var]) 
print "\nAccuracy : %s" % "{0:.4%}".format(accuracy)
pred1 = pred[ pred[outcome_var] == 1 ]
pred0 = pred[ pred[outcome_var] == 0 ]
accuracy = metrics.accuracy_score(pred1[predicted_var],pred1[outcome_var]) 
print "Accuracy 1 : %s" % "{0:.4%}".format(accuracy)
accuracy = metrics.accuracy_score(pred0[predicted_var],pred0[outcome_var]) 
print "Accuracy 0 : %s" % "{0:.4%}".format(accuracy)

pred = pd.read_csv("myosotis_database_prediction_linux_3.csv") 
accuracy = metrics.accuracy_score(pred[predicted_var],pred[outcome_var]) 
print "\nAccuracy : %s" % "{0:.4%}".format(accuracy)
pred1 = pred[ pred[outcome_var] == 1 ]
pred0 = pred[ pred[outcome_var] == 0 ]
accuracy = metrics.accuracy_score(pred1[predicted_var],pred1[outcome_var]) 
print "Accuracy 1 : %s" % "{0:.4%}".format(accuracy)
accuracy = metrics.accuracy_score(pred0[predicted_var],pred0[outcome_var]) 
print "Accuracy 0 : %s" % "{0:.4%}".format(accuracy)

pred = pd.read_csv("myosotis_database_prediction_linux_4.csv") 
accuracy = metrics.accuracy_score(pred[predicted_var],pred[outcome_var]) 
print "\nAccuracy : %s" % "{0:.4%}".format(accuracy)
pred1 = pred[ pred[outcome_var] == 1 ]
pred0 = pred[ pred[outcome_var] == 0 ]
accuracy = metrics.accuracy_score(pred1[predicted_var],pred1[outcome_var]) 
print "Accuracy 1 : %s" % "{0:.4%}".format(accuracy)
accuracy = metrics.accuracy_score(pred0[predicted_var],pred0[outcome_var]) 
print "Accuracy 0 : %s" % "{0:.4%}".format(accuracy)

