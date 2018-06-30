# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:26:26 2016

@author: Carlos
"""

import pandas as pd
import unicodedata
import nltk

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def dprint(texto):
    print ('<---| ', texto, ' |--->')

#Definindo uma funcao de listagem e descricao do dataframe
def desc_df(df,n):
    print '\nData File data and stats'
    print df.head(n)
    print df.describe()
    print df.info()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def is_string(s):
    try:
        str(s)
        return True
    except ValueError:
        return False
def clean_string(s):
    l_rep = [',','.','\\t','\\s','\\n','\\r','  ','   ','    ','     ','      ']
    l_stop = ['aparentemente','desaparecimento','motivado','sobre','acontecido','','','','']
    for i in l_rep:
        s = s.replace(i, ' ')
    if not is_string(row['informacoes']):
        s = unicodedata.normalize('NFKD', s).encode('ascii','ignore')
        print s

        l_s = s.lower().split()
        l_s1 = []
        for i in range(len(l_s)):
            if not l_s[i] in nltk.corpus.stopwords.words('portuguese'):
               l_s1.append(l_s[i])
        s = ' '.join(l_s1)

        l_s = s.lower().split()
        l_s1 = []
        for i in range(len(l_s)):
            if not l_s[i] in l_stop:
               l_s1.append(l_s[i])
        s = ' '.join(l_s1)

        print s
    s = s.upper()
    return s
    

# Runnning instructions 
print '='*200
print '\nStarted'

#Reading the dataset in a dataframe using Pandas
df_full = pd.read_excel("myosotis_database.xlsx") 

print "df_full File"
#desc_df(df_full,5)

#print df_full[['marca_caracteristica','status','id']].groupby(['marca_caracteristica','status']).count()


dprint('informacoes Cleanup')

x = ''
for index, row in df_full.iterrows():
    if is_number(row['informacoes']):
        df_full.at[index, 'informacoes'] = 'UNDEFINED'
    else:
        x = x + ' ' + clean_string(row['informacoes'])
        df_full.at[index, 'informacoes'] = clean_string(row['informacoes'])
    

df_full['informacoes'] = df_full['informacoes'].astype(str)
print df_full[['informacoes','status','id']].groupby(['informacoes','status']).count()
print x

"""
#print df_full[['informacoes','status','id']].groupby(['informacoes','status']).count()
for index, row in df_full.iterrows():
    if is_number(row['informacoes']):
        df_full.at[index, 'informacoes'] = 'UNDEFINED'
    elif is_string(row['informacoes']):
        df_full.at[index, 'informacoes'] = row['informacoes'].replace('\\t', ' ')
        df_full.at[index, 'informacoes'] = row['informacoes'].replace('\\T', ' ')
        df_full.at[index, 'informacoes'] = row['informacoes'].upper()
    else:
        df_full.at[index, 'informacoes'] = unicodedata.normalize('NFKD', row['informacoes']).encode('ascii','ignore')
        df_full.at[index, 'informacoes'] = row['informacoes'].replace('\\t', ' ')
        df_full.at[index, 'informacoes'] = row['informacoes'].replace('\\T', ' ')
        df_full.at[index, 'informacoes'] = row['informacoes'].upper()
    

#df_full['informacoes'] = df_full['informacoes'].astype(str)
print df_full[['informacoes','status','id']].groupby(['informacoes','status']).count()

"""
