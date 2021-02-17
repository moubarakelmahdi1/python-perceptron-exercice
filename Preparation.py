# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:04:05 2020

@author: estel
"""

"""
Projet IDS 
Duhem Estelle - Moubarak El Mahdi 
INGE1 (L3) - Fi22
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def ValuesMissing(data):
    """
    Fonction qui permet de détecter les valeurs manquantes
    """
    
    missingValues = []
    for column,value in data.iteritems(): 
        for i in value:
            if pd.isnull(i):
                missingValues.append(i)
    return missingValues 

def replaceValuesMissing(data):
    """
    Fonction qui permet de remplacer les valeurs manquantes
    par la moyenne de l'attribut concerné
    """
    
    for column,value in data.iteritems(): 
        m = data[str(column)].mean()
        data[str(column)].replace(np.nan,m,inplace=True)
    return data 

def ValuesNotNumbers(data):
    """
    Cette fonction permet de détecter les données
    qui sont non-numériques
    """
    
    notNumbersValues = []
    for e in data:
        try:
            print (float('DataCamp'))
        except ValueError:
            notNumbersValues.append(e)
    return notNumbersValues


def replaceAbnormalValues(data):
    """
    Fonction qui permet de détecter les données aberrantes 
    avec le critere des trois sigmas
    sur les attributs qui semblent suivre une loi normale
    et qui remplace ces données par la moyenne de l'attribut concerné
    """
    
    j=0
    temp = data.copy()
    for column,value in temp.iteritems(): 
        m = data[str(column)].mean()
        e = data[str(column)].std()
        for i in value:
            if (i>m+3*e or i<m-3*e) & (str(column)=='fixed acidity' or str(column)=='volatile acidity' or str(column)=='density' or str(column)=='pH' ):
                temp[temp==i]=m
                j=j+1
    print ('Nombre de données aberrantes : ' + str(j))
    return temp
  

def deleteAbnormalValues(data):
    """
    Fonction qui permet de détecter les données aberrantes
    avec le critere des trois sigmas
    sur les attributs qui semblent suivre une loi normale
    et qui supprime ces données
    """
    
    temp = data.copy()
    for column,value in temp.iteritems(): 
        m = data[str(column)].mean()
        e = data[str(column)].std()
        for i in value:
            if (i>m+3*e or i<m-3*e) & (str(column)=='fixed acidity' or str(column)=='volatile acidity' or str(column)=='density' or str(column)=='pH' ):
                temp = temp.drop(temp[temp[str(column)]== i].index)
    return temp 

def graphiquesDistributionsAttributs(replacedData):
    """
    Fonction qui permet d'afficher les 4 figures représentant
    les distributions des données pour chaque attribut 
    regroupées 3 par 3
    """
    
    figA,axA = plt.subplots(3)
    axA[0].hist(replacedData['fixed acidity'],color = "pink", label = "fixed acidity", histtype='stepfilled', bins=120)
    axA[0].legend()
    axA[1].hist(replacedData['volatile acidity'],color = "grey", label = "volatile acidity", histtype='stepfilled', bins=120)
    axA[1].legend()
    axA[2].hist(replacedData['citric acid'],color = "purple", label = "citric acid", histtype='stepfilled', bins=120)
    axA[2].legend()
    
    figB,axB = plt.subplots(3)
    axB[0].hist(replacedData['residual sugar'],color = "pink", label = "residual sugar", histtype='stepfilled', bins=120)
    axB[0].legend()
    axB[1].hist(replacedData['chlorides'],color = "grey", label = "chlorides")
    axB[1].legend()
    axB[2].hist(replacedData['free sulfur dioxide'],color = "purple", label = "free sulfur dioxide", histtype='stepfilled', bins=120)
    axB[2].legend()
    
    figC,axC = plt.subplots(3)
    axC[0].hist(replacedData['total sulfur dioxide'],color = "pink", label = "total sulfur dioxide", histtype='stepfilled', bins=120)
    axC[0].legend()
    axC[1].hist(replacedData['density'],color = "grey", label = "density")
    axC[1].legend()
    axC[2].hist(replacedData['pH'],color = "purple", label = "pH", histtype='stepfilled', bins=120)
    axC[2].axis([2.5,4.5,0,400])
    axC[2].legend()
    
    figD,axD = plt.subplots(3)
    axD[0].hist(replacedData['sulphates'],color = "pink", label = "sulphates", histtype='stepfilled', bins=120)
    axD[0].legend()
    axD[1].hist(replacedData['alcohol'],color = "grey", label = "alcohol", histtype='stepfilled', bins=120)
    axD[1].legend()
    axD[2].hist(replacedData['quality'],color = "purple", label = "quality", histtype='stepfilled', bins=120)
    axD[2].legend()

def centrer_reduire(replacedData):
    """
    Cette fonction permet de centrer et reduire nos données
    """
    
    for columns in replacedData.columns:
        if columns != 'quality':
            replacedData[columns] = replacedData[columns].sub(replacedData[columns].mean())
            replacedData[columns] = replacedData[columns].div(replacedData[columns].std())
    return replacedData