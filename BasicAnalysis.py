# -*- coding: utf-8 -*-
'''
Created on Wed Nov 16 21:02:24 2016

@author: Carlos
'''

# Imports

# pandas
import pandas as pd
import numpy as np

# matplotlib, seaborn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

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

print '\nRead CSV'
#Reading the dataset in a dataframe using Pandas
df_full = pd.read_csv('myosotis_database_encoded.csv') 

print '\ndf_full File'
desc_df(df_full,5)
# status id nome sexo olhos cor_da_pele cabelo peso_aproximado altura_aproximada 
# tipo_fisico transtorno_mental  idade data_nascimento  dias_desaparecido 
# data_desaparecimento bairro_desaparecimento cidade_desaparecimento uf_desaparecimento  anos_desaparecido

#iolhos icor_da_pele icabelo  peso_aproximado  altura_aproximada  itipo_fisico  itranstorno_mental
#idade  dias_desaparecido  anos_desaparecido    dias_round  idade_round   peso_round  altura_round

# E = 1, D = 0
print ('Map status')
df_full['istatus'] = df_full['status'].map( {'D': 0, 'E': 1} ).astype(int)
"""

# plot
print df_full[['transtorno_mental','status','id']].groupby(['transtorno_mental','status']).count()
print df_full[['uf_desaparecimento','status','id']].groupby(['uf_desaparecimento','status']).count()
print df_full[['sexo','status','id']].groupby(['sexo','status']).count()
print df_full[['idade','status','id']].groupby(['idade','status']).count()
print df_full[['dias_desaparecido','status','id']].groupby(['dias_desaparecido','status']).count()
print df_full[['peso_aproximado','id']].groupby(['peso_aproximado']).count()
print df_full[['altura_aproximada','id']].groupby(['altura_aproximada']).count()

sns.factorplot('transtorno_mental','istatus', data=df_full,size=4,aspect=3)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='transtorno_mental', data=df_full, ax=axis1)
sns.countplot(x='istatus', hue='transtorno_mental', data=df_full, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked
transtorno_mental_perc = df_full[['transtorno_mental', 'istatus']].groupby(['transtorno_mental'],as_index=False).mean()
sns.barplot(x='transtorno_mental', y='istatus', data=transtorno_mental_perc,ax=axis3) #,order=['S','C','Q']
"""

print df_full[['istatus','id']].groupby(['istatus']).count()

#Figure 1
data = df_full[['istatus','id']][(df_full['dias_round']>1450) & (df_full['dias_round']<4550) & (df_full['itranstorno_mental']==1) & (df_full['altura_round']<145)].groupby(['istatus']).count()
print data
data = df_full[['istatus','id']][(df_full['dias_round']>1450) & (df_full['dias_round']<4550) & (df_full['itranstorno_mental']==1) & (df_full['altura_round']>=145)].groupby(['istatus']).count()
print data

data = df_full[['istatus','id']][ (df_full['itranstorno_mental']==1) & (df_full['altura_round']<145)].groupby(['istatus']).count()
print data
data = df_full[['istatus','id']][ (df_full['itranstorno_mental']==1) & (df_full['altura_round']>=145)].groupby(['istatus']).count()
print data

#Figure 2
data = df_full[['istatus','id']][(df_full['itranstorno_mental']==1) & (df_full['idade_round']<31.5) & (df_full['peso_round']<57.5) & (df_full['isexo']==0)].groupby(['istatus']).count()
print data
data = df_full[['istatus','id']][(df_full['itranstorno_mental']==1) & (df_full['idade_round']<31.5) & (df_full['peso_round']<57.5) & (df_full['isexo']==1)].groupby(['istatus']).count()
print data

#Figure 3
print df_full[['iolhos','istatus','id']][(df_full['itranstorno_mental']==1) & (df_full['dias_round']<4900) & (df_full['idade_round']>19.5)].groupby(['iolhos','istatus']).count()


desc_df(data,6)
x = data[['dias_desaparecido']].values.copy()
y1 = data[['altura_aproximada']].values.copy()
y2 = data[['istatus']].values.copy()

# red dashes, blue squares and green triangles
plt.plot(x, y1, 'g^',x, y2, 'bs')#, 'r--')#, x, y2, 'bs') #, t, t**3, 'g^')
plt.show()