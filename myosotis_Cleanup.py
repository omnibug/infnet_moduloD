# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:13:31 2016

@author: Carlos
"""
import pandas as pd
import numpy as np
import unicodedata
import json
import math
import datetime as dt

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

def fl_stepper_1(n,step):
    s = round(n/step,0)*step
    #    print n, s
    return s

# Runnning instructions 
print '='*200
print '\nStarted'

#Reading the dataset in a dataframe using Pandas
df_full = pd.read_excel("myosotis_database.xlsx") 

print "\ndf_fullFile"
desc_df(df_full,5)

dprint('id Cleanup')
df_full = df_full[df_full['nome'].notnull()]
df_full['id'] = df_full['id'].astype(int)

dprint('nome Cleanup')
for index, row in df_full.iterrows():
    df_full.at[index, 'nome'] = unicodedata.normalize('NFKD', row['nome']).encode('ascii','ignore')
df_full['nome'] = df_full['nome'].astype(str)

dprint('Sex Cleanup')
#print df_full.groupby('sexo').count()
df_full['sexo'] = df_full['sexo'].astype(str)
df_full['sexo'] = df_full['sexo'].map( {'Feminino': 'F', 'Masculino': 'M', 'F': 'F', 'M':'M'} ).astype(str)
df_full['id'] = df_full['id'].astype(str)
l_names_null=[]
for index, row in df_full.iterrows():
    if row['sexo'] == 'nan': #pd.isnull(row['sexo']):
        l_names_null.append(row['nome'].split()[0])
l_names_null = list(set(l_names_null))
#print l_names_null
#[u'moria', u'igor', u'luana', u'daiane', u'aerte', u'jose', u'pedro', u'tamires', u'vitoria', u'talita', u'cintia', u'gisele', u'deivid', u'bruna', u'paula', u'kelen', u'andrielly', u'jhontan', u'n\xedcolas', u'schailen', u'mauren', u'shayane', u'larissa', u'augusto', u'alex', u'barbara', u'alice', u'jenifer', u'rafaella', u'jean', u'amanda']
l_F = ['moria', 'luana', 'daiane', 'tamires', 'vitoria', 'talita', 'cintia', 'gisele', 'bruna', 'paula', 'kelen', 'andrielly', 'schailen', 'mauren', 'shayane', 'larissa', 'barbara', 'alice', 'jenifer', 'rafaella', 'amanda'] 
for index, row in df_full.iterrows():
    if row['sexo'] == 'nan':
        for i in range(len(l_F)):
            if row['nome'].split()[0] == l_F[i]:
                df_full.at[index, 'sexo'] = 'F'
            else:
                df_full.at[index, 'sexo'] = 'M'      
#print df_full[['sexo','id']].groupby('sexo').count()

dprint('olhos Cleanup')
#print df_full.groupby('olhos').count()
l_olhos_cor = []
l_castanho = ['castanho','castanhos','mel']
l_preto = ['preto','pretos']
l_azul = ['azul','azuis']
l_verde = ['verde','verdes']
for index, row in df_full.iterrows():
    if pd.isnull(row['olhos']) is False:
#        print row['olhos']
        l_olhos_cor = row['olhos'].lower().split()
        for i in range(len(l_olhos_cor)):
            if any(l_olhos_cor[i] in s for s in l_castanho):
               df_full.at[index, 'olhos'] = 'castanho'
            elif any(l_olhos_cor[i] in s for s in l_preto):
               df_full.at[index, 'olhos'] = 'preto'
            elif any(l_olhos_cor[i] in s for s in l_azul):
               df_full.at[index, 'olhos'] = 'azul'
            elif any(l_olhos_cor[i] in s for s in l_verde):
               df_full.at[index, 'olhos'] = 'verde'
            else:
               df_full.at[index, 'olhos'] = 'UNDEFINED'
    else:
        df_full.at[index, 'olhos'] = 'UNDEFINED'      
df_full['olhos'] = df_full['olhos'].astype(str)
#print df_full.groupby('olhos').count()

dprint('cor_da_pele Cleanup')
#print df_full.groupby('cor_da_pele').count()
l_cor_da_pele_cor = []
l_branco = ['branco','branca']
l_amarelo = ['amarelo','amarela']
l_mulato = ['mulato','moreno','morena']
l_negro = ['negro','negra']
l_pardo = ['pardo','parda']
l_indigena = ['indigena']
for index, row in df_full.iterrows():
    if pd.isnull(row['cor_da_pele']) is False:
        l_cor_da_pele_cor = row['cor_da_pele'].lower().split()
        for i in range(len(l_cor_da_pele_cor)):
            if any(l_cor_da_pele_cor[i] in s for s in l_branco):
               df_full.at[index, 'cor_da_pele'] = 'branco'
            elif any(l_cor_da_pele_cor[i] in s for s in l_amarelo):
               df_full.at[index, 'cor_da_pele'] = 'amarelo'
            elif any(l_cor_da_pele_cor[i] in s for s in l_mulato):
               df_full.at[index, 'cor_da_pele'] = 'mulato'
            elif any(l_cor_da_pele_cor[i] in s for s in l_negro):
               df_full.at[index, 'cor_da_pele'] = 'negro'
            elif any(l_cor_da_pele_cor[i] in s for s in l_pardo):
               df_full.at[index, 'cor_da_pele'] = 'pardo'
            elif any(l_cor_da_pele_cor[i] in s for s in l_indigena):
               df_full.at[index, 'cor_da_pele'] = 'indigena'
            else:
               df_full.at[index, 'cor_da_pele'] = 'UNDEFINED'
    else:
        df_full.at[index, 'cor_da_pele'] = 'UNDEFINED'      
df_full['cor_da_pele'] = df_full['cor_da_pele'].astype(str)
#print df_full.groupby('cor_da_pele').count()

dprint('cabelo Cleanup')
#print df_full.groupby('cabelo').count()
l_cor = []
l_branco = ['branco','branca','grisalho']
l_castanho = ['castanho','castanhogrisalho']
l_careca = ['careca']
l_preto = ['preto']
l_loiro = ['loiro']
l_ruivo = ['ruivo','canela']
for index, row in df_full.iterrows():
    if pd.isnull(row['cabelo']) is False:
        l_cor = row['cabelo'].lower().split()
        for i in range(len(l_cor)):
            if any(l_cor[i] in s for s in l_branco):
               df_full.at[index, 'cabelo'] = 'branco'
            elif any(l_cor[i] in s for s in l_castanho):
               df_full.at[index, 'cabelo'] = 'castanho'
            elif any(l_cor[i] in s for s in l_careca):
               df_full.at[index, 'cabelo'] = 'careca'
            elif any(l_cor[i] in s for s in l_preto):
               df_full.at[index, 'cabelo'] = 'preto'
            elif any(l_cor[i] in s for s in l_loiro):
               df_full.at[index, 'cabelo'] = 'loiro'
            elif any(l_cor[i] in s for s in l_ruivo):
               df_full.at[index, 'cabelo'] = 'ruivo'
            else:
               df_full.at[index, 'cabelo'] = 'UNDEFINED'
    else:
        df_full.at[index, 'cabelo'] = 'UNDEFINED'      
df_full['cabelo'] = df_full['cabelo'].astype(str)
#print df_full.groupby('cabelo').count()

dprint('peso_aproximado Cleanup')
#print df_full.groupby('peso_aproximado').count()
l_peso = []
for index, row in df_full.iterrows():
    if pd.isnull(row['peso_aproximado']) is False:
        if is_number(row['peso_aproximado']):
           df_full.at[index, 'peso_aproximado'] = float(row['peso_aproximado'])
        else:
            l = list(row['peso_aproximado'])
            for i in range(len(l)):
                if l[i] == ',':
                    l[i] = '.'
            s = ''.join(l)
            l_peso = s.lower().split()
            for i in range(len(l_peso)):
                f_peso = 0.0
                if is_number(l_peso[i]):
                    f_peso = float(l_peso[i])
                    df_full.at[index, 'peso_aproximado'] = float(l_peso[i])
                elif f_peso == 0.0:
                    df_full.at[index, 'peso_aproximado'] = 0.0
    else:
        df_full.at[index, 'peso_aproximado'] = 0.0

for index, row in df_full[df_full['peso_aproximado']>1000].iterrows():
    df_full.at[index, 'peso_aproximado'] = df_full.at[index, 'peso_aproximado'] / 1000
df_full['peso_aproximado'] = df_full['peso_aproximado'].astype(float)
#print df_full[['peso_aproximado','status']].groupby('peso_aproximado').count()

dprint('altura_aproximada Cleanup')
#print df_full.groupby('altura_aproximada').count()
l_peso = []
for index, row in df_full.iterrows():
    if pd.isnull(row['altura_aproximada']) is False:
        if is_number(row['altura_aproximada']):
           df_full.at[index, 'altura_aproximada'] = float(row['altura_aproximada'])
        else:
            l = list(row['altura_aproximada'])
            for i in range(len(l)):
                if l[i] == ',':
                    l[i] = '.'
            s = ''.join(l)
            l_peso = s.lower().split()
            for i in range(len(l_peso)):
                f_peso = 0.0
                if is_number(l_peso[i]):
                    f_peso = float(l_peso[i])
                    df_full.at[index, 'altura_aproximada'] = float(l_peso[i])
                elif f_peso == 0.0:
                    df_full.at[index, 'altura_aproximada'] = 0.0
    else:
        df_full.at[index, 'altura_aproximada'] = 0.0
for index, row in df_full[df_full['altura_aproximada']>3].iterrows():
    df_full.at[index, 'altura_aproximada'] = df_full.at[index, 'altura_aproximada'] / 100
df_full['altura_aproximada'] = df_full['altura_aproximada'].astype(float)
#print df_full[['altura_aproximada','id']].groupby('altura_aproximada').count()

dprint('idade Cleanup')
#print df_full.groupby('idade').count()
l_peso = []
for index, row in df_full.iterrows():
    if pd.isnull(row['idade']) is False:
        if is_number(row['idade']):
           df_full.at[index, 'idade'] = float(row['idade'])
        else:
            l = list(row['idade'])
            for i in range(len(l)):
                if l[i] == ',':
                    l[i] = '.'
            s = ''.join(l)
            l_peso = s.lower().split()
            for i in range(len(l_peso)):
                if is_number(l_peso[i]):
                    df_full.at[index, 'idade'] = float(l_peso[i])
 #               else:
 #                   df_full.at[index, 'idade'] = 0.0
    else:
        df_full.at[index, 'idade'] = 0.0

for index, row in df_full[df_full['idade']>100000].iterrows():
    df_full.at[index, 'idade'] = df_full.at[index, 'idade'] / 1000
df_full['idade'] = df_full['idade'].astype(float)
#print df_full[['idade','status']].groupby('idade').count()

dprint('idade_round ajustment')
df_full['idade_round'] =  df_full['idade'].astype(int)
for index, row in df_full.iterrows():
    df_full.at[index, 'idade_round'] = fl_stepper_1(row['idade'], 3)
print df_full[['idade_round','status']].groupby('idade_round').count()

#estimar altura/peso com base na idade
#estimar altura/peso com base no imc medio
#IMC = peso / (altura x altura).
imc = (18.5 + 24.9)/2
dprint('altura_round ajustment')
df_full['altura_round'] =  df_full['altura_aproximada'].astype(int)
for index, row in df_full.iterrows():
    f_altura = row['altura_aproximada']
    if f_altura == 0.0:
        f_altura = math.sqrt( row['peso_aproximado'] / imc )
    if f_altura == 0.0:
        if row['idade'] <= 1:
            f_altura = .815
        elif row['idade'] <= 2:
            f_altura = 0.92
        elif row['idade'] <= 3:
            f_altura = 0.989
        elif row['idade'] <= 4:
            f_altura = 1.062
        elif row['idade'] <= 5:
            f_altura = 1.120
        elif row['idade'] <= 6:
            f_altura = 1.183
        elif row['idade'] <= 7:
            f_altura = 1.249
        elif row['idade'] <= 8:
            f_altura = 1.297
        elif row['idade'] <= 9:
            f_altura = 1.352
        elif row['idade'] <= 10:
            f_altura = 1.399
        elif row['idade'] <= 11:
            f_altura = 1.436
        elif row['idade'] <= 12:
            f_altura = 1.510
        elif row['idade'] <= 13:
            f_altura = 1.575
        elif row['idade'] <= 14:
            f_altura = 1.641
        elif row['idade'] <= 15:
            f_altura = 1.678
        elif row['idade'] <= 16:
            f_altura = 1.700
        elif row['idade'] <= 17:
            f_altura = 1.718
        elif row['idade'] <= 18:
            f_altura = 1.726
        elif row['idade'] <= 19:
            f_altura = 1.720
        elif row['idade'] <= 24:
            f_altura = 1.730
        elif row['idade'] <= 29:
            f_altura = 1.730
        elif row['idade'] <= 34:
            f_altura = 1.716
        elif row['idade'] <= 44:
            f_altura = 1.710
        elif row['idade'] <= 54:
            f_altura = 1.699
        elif row['idade'] <= 64:
            f_altura = 1.682
        elif row['idade'] <= 74:
            f_altura = 1.669
        else:
            f_altura = 1.657
            
    df_full.at[index, 'altura_round'] = fl_stepper_1(f_altura*100, 10)
print df_full[['altura_round','status']].groupby('altura_round').count()

dprint('peso_round ajustment')
df_full['peso_round'] =  df_full['peso_aproximado'].astype(int)
for index, row in df_full.iterrows():
    f_peso = row['peso_aproximado']
    if f_peso == 0.0 and row['altura_aproximada']  <> 0.0:
        f_peso = math.sqrt( imc / row['altura_aproximada'] )
    df_full.at[index, 'peso_round'] = fl_stepper_1(f_peso, 5)
print df_full[['peso_round','status']].groupby('peso_round').count()


dprint('tipo_fisico Cleanup')
#print df_full.groupby('tipo_fisico').count()
for index, row in df_full.iterrows():
    if pd.isnull(row['tipo_fisico']) is False:
        if row['tipo_fisico'].lower() == u'raquítico':
            df_full.at[index, 'tipo_fisico'] = 'magro'
    else:
        df_full.at[index, 'tipo_fisico'] = 'UNDEFINED'      
df_full['tipo_fisico'] = df_full['tipo_fisico'].astype(str)
#print df_full.groupby('tipo_fisico').count()

dprint('transtorno_mental Cleanup')
#print df_full.groupby('transtorno_mental').count()
for index, row in df_full.iterrows():
    if pd.isnull(row['transtorno_mental']) is False:
        if row['transtorno_mental'].lower() == u'não':
            df_full.at[index, 'transtorno_mental'] = 'NAO'
    else:
        df_full.at[index, 'transtorno_mental'] = 'UNDEFINED'
df_full['transtorno_mental'] = df_full['transtorno_mental'].astype(str)
#print df_full.groupby('transtorno_mental').count()

dprint('data_nascimento Cleanup')
#print df_full.groupby('data_nascimento').count()
for index, row in df_full.iterrows():
    if type(row['data_nascimento']) is not pd.tslib.Timestamp:
        df_full.at[index, 'data_nascimento'] = pd.to_datetime(row['data_nascimento'], format='%d/%m/%Y', errors='ignore')
for index, row in df_full.iterrows():
    if type(row['data_nascimento']) is not pd.tslib.Timestamp:
        df_full.at[index, 'data_nascimento'] = pd.to_datetime(row['data_nascimento'], format='%Y-%m-%d', errors='ignore')
for index, row in df_full.iterrows():
    if type(row['data_nascimento']) is not pd.tslib.Timestamp:
        df_full.at[index, 'data_nascimento'] = pd.to_datetime(row['data_nascimento'], format='%Y-%m-%d %H:%T:%S', errors='ignore')
for index, row in df_full.iterrows():
    if type(row['data_nascimento']) is not pd.tslib.Timestamp:
        if type(row['data_nascimento']) is not pd.tslib.NaTType:
            print (row['data_nascimento'], type(row['data_nascimento']))
df_full['data_nascimento'] = pd.to_datetime(df_full['data_nascimento'])

dprint('dias_desaparecido Cleanup')
l_peso = []
for index, row in df_full.iterrows():
    if pd.isnull(row['dias_desaparecido']) is False:
        if is_number(row['dias_desaparecido']):
           df_full.at[index, 'dias_desaparecido'] = float(row['dias_desaparecido'])
        else:
            dias = unicodedata.normalize('NFKD', row['dias_desaparecido']).encode('ascii','ignore')
            #if is_string(dias) is False:
            dias = str(dias)
            s = dias.replace('\\t', ' ')
            l_peso = s.lower().split()
            for i in range(len(l_peso)):
                if is_number(l_peso[i]):
                    df_full.at[index, 'dias_desaparecido'] = float(l_peso[i])
#                else:
#                    df_full.at[index, 'dias_desaparecido'] = 0.0
    else:
        df_full.at[index, 'dias_desaparecido'] = 0.0
df_full['dias_desaparecido'] = df_full['dias_desaparecido'].astype(float)
#print df_full[['dias_desaparecido','status']].groupby('dias_desaparecido').count()

dprint('data_desaparecimento Cleanup')
#print df_full.groupby('data_desaparecimento').count()
for index, row in df_full.iterrows():
    if type(row['data_desaparecimento']) is not pd.tslib.Timestamp:
        df_full.at[index, 'data_desaparecimento'] = pd.to_datetime(row['data_desaparecimento'], format='%d/%m/%Y', errors='ignore')
for index, row in df_full.iterrows():
    if type(row['data_desaparecimento']) is not pd.tslib.Timestamp:
        df_full.at[index, 'data_desaparecimento'] = pd.to_datetime(row['data_desaparecimento'], format='%Y-%m-%d', errors='ignore')
for index, row in df_full.iterrows():
    if type(row['data_desaparecimento']) is not pd.tslib.Timestamp:
        df_full.at[index, 'data_desaparecimento'] = pd.to_datetime(row['data_desaparecimento'], format='%Y-%m-%d %H:%T:%S', errors='ignore')
for index, row in df_full.iterrows():
    if type(row['data_desaparecimento']) is not pd.tslib.Timestamp:
        if type(row['data_desaparecimento']) is not pd.tslib.NaTType:
            print (row['data_desaparecimento'], type(row['data_desaparecimento']))
df_full['data_desaparecimento'] = pd.to_datetime(df_full['data_desaparecimento'])
#print df_full[['data_desaparecimento','id']].groupby('data_desaparecimento').count()

dprint('anos_desaparecido Cleanup')
#print df_full.groupby('anos_desaparecido').count()
for index, row in df_full.iterrows():
    if pd.isnull(row['anos_desaparecido']) is False:
        if is_number(row['anos_desaparecido']):
           df_full.at[index, 'anos_desaparecido'] = float(row['anos_desaparecido'])
        else:
           df_full.at[index, 'anos_desaparecido'] = 0.0
    else:
        df_full.at[index, 'anos_desaparecido'] = 0.0
df_full['anos_desaparecido'] = df_full['anos_desaparecido'].astype(float)
#print df_full[['anos_desaparecido','status']].groupby('anos_desaparecido').count()

def fl_stepper(n):
#    if n < 100:
#        s = round(n,-1)
    if n < 1000:
        s = fl_stepper_1(n,50)
    elif n < 6000:
        s = fl_stepper_1(n,100)
    elif n < 10000:
        s = fl_stepper_1(n,1000)
    elif n < 100000:
        s = fl_stepper_1(n,10000)
    else:
        s = fl_stepper_1(n,100000)
    return s

dprint('dias_round ajustment')
df_full['dias_round'] =  df_full['dias_desaparecido'].astype(int)
data_corrente = dt.datetime.now()
for index, row in df_full.iterrows():
    dias_round = row['dias_round']
    if dias_round == 0.0 and row['data_desaparecimento'] is np.datetime64 and (row['status'] == 'Desaparecido(a)' or row['status'] == 'D' ):
        print data_corrente - row['data_desaparecimento']
    if dias_round == 0.0 and row['anos_desaparecido'] <> 0.0:
        dias_round = row['anos_desaparecido'] * 365.25
    df_full.at[index, 'dias_round'] = fl_stepper(dias_round)
x = df_full[['dias_round','status']].groupby('dias_round').count()
print x
x.plot(figsize=(10, 6))

dprint('bairro_desaparecimento Cleanup')
#print df_full.groupby('bairro_desaparecimento').count()
for index, row in df_full.iterrows():
    if pd.isnull(row['bairro_desaparecimento']) is False:
            bairro = unicodedata.normalize('NFKD', row['bairro_desaparecimento']).encode('ascii','ignore')
            df_full.at[index, 'bairro_desaparecimento'] =  bairro.lower()
    else:
       df_full.at[index, 'bairro_desaparecimento'] = 'UNDEFINED'
#print df_full.groupby('bairro_desaparecimento').count()

dprint('cidade_desaparecimento Cleanup')
#print df_full.groupby('cidade_desaparecimento').count()
for index, row in df_full.iterrows():
    if is_number(row['cidade_desaparecimento']) is True:
       df_full.at[index, 'cidade_desaparecimento'] = 'UNDEFINED'
    elif pd.isnull(row['cidade_desaparecimento']) is False:
            bairro = unicodedata.normalize('NFKD', row['cidade_desaparecimento']).encode('ascii','ignore')
            df_full.at[index, 'cidade_desaparecimento'] =  bairro.lower()
    else:
       df_full.at[index, 'cidade_desaparecimento'] = 'UNDEFINED'
#print df_full.groupby('cidade_desaparecimento').count()

dprint('uf_desaparecimento Cleanup')
#print df_full.groupby('uf_desaparecimento').count()
for index, row in df_full.iterrows():
    if is_number(row['uf_desaparecimento']) is True:
       df_full.at[index, 'uf_desaparecimento'] = 'UNDEFINED'
    elif pd.isnull(row['uf_desaparecimento']) is False:
            bairro = unicodedata.normalize('NFKD', row['uf_desaparecimento']).encode('ascii','ignore')
            df_full.at[index, 'uf_desaparecimento'] =  bairro.upper()
    else:
       df_full.at[index, 'uf_desaparecimento'] = 'UNDEFINED'
#print df_full.groupby('uf_desaparecimento').count()

dprint('status Cleanup')
#print df_full.groupby('status').count()
df_full['status'] = df_full['status'].astype(str)
df_full['status'] = df_full['status'].map( {'Desaparecido(a)': 'D', 'Encontrado(a)': 'E'} ).astype(str)
#print df_full[['status','id']].groupby('status').count()

#Writing the dataset from a dataframe using Pandas
dprint('Write Cleaned File')
c_columns = ['status','id','nome','sexo','olhos','cor_da_pele','cabelo','peso_aproximado','altura_aproximada','tipo_fisico','transtorno_mental','idade','data_nascimento','dias_desaparecido','data_desaparecimento','bairro_desaparecimento','cidade_desaparecimento','uf_desaparecimento','anos_desaparecido','dias_round']
df_full[c_columns].to_csv('myosotis_database_clean.csv',index=False) 

#desc_df(df_full[c_columns],5)

dprint('Numeric Encoder')
l_fields = ['sexo','olhos','cor_da_pele','cabelo','tipo_fisico','transtorno_mental','idade','data_nascimento','dias_desaparecido','data_desaparecimento','status','anos_desaparecido','peso_round','altura_round','dias_round','idade_round']
l_list = []
for i in range(len(l_fields)):
    s_type = df_full[l_fields[i]].dtype
    print l_fields[i], s_type
    if s_type == 'object':
        l_values = list(enumerate(np.unique(df_full[l_fields[i]])))         # determine all values for column
        d_values = { name : i for i, name in l_values }             # set up a dictionary in the form  values : index
        l_list.append([l_fields[i],l_values])
        print d_values
        df_full['i'+l_fields[i]] = df_full[l_fields[i]].map( lambda x: d_values[x]).astype(int)     # Convert all Title strings to int
with open('encoded_data.json', 'w') as f:
    f.write(json.dumps(l_list))
f.closed

#Writing the dataset from a dataframe using Pandas
dprint('Write Encoded File')
e_columns = ['istatus','id','nome','isexo','iolhos','icor_da_pele','icabelo','peso_aproximado','altura_aproximada','itipo_fisico','itranstorno_mental','idade','data_nascimento','dias_desaparecido','data_desaparecimento','anos_desaparecido','dias_round','idade_round','peso_round','altura_round']
df_full[e_columns].to_csv('myosotis_database_encoded.csv',index=False) 

#desc_df(df_full[e_columns],5)
