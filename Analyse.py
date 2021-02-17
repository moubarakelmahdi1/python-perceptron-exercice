# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:49:26 2020

@author: estel
"""

"""
Projet IDS 
Duhem Estelle - Moubarak El Mahdi 
INGE1 (L3) - Fi22
"""

import matplotlib.pyplot as plt



def allQualityValues(data):
    """
    Cette fonction permet de savoir quelles sont les valeurs
    prisent par l'attribut quality.
    Elle retourne une liste avec toutes les valeurs possibles.
    """
    
    qualityValues = []
    for e in data['quality']:
        if (e in qualityValues) == False:
            qualityValues.append(e)
            
    return qualityValues
        

#Voici nos 2 classes
def graphiquesRepartitionClasses(data):
    """
    Cette fonction permet d'afficher les 12 graphiques avec la répartition
    des qualités en fonction de chaque attribut 
    """
    
    
    dataBestQuality = data[data['quality']==1]
    dataWorstQuality = data[data['quality']==-1]

    colors = ["blue", "red"]
    labels = ["Best Quality", "Worst Quality"]

    fig1,ax1 = plt.subplots(2)

    ax1[0].plot(dataWorstQuality['fixed acidity'],'+r',label='fixed acidity and worst quality')
    ax1[0].plot(dataBestQuality['fixed acidity'],'+b',label='fixed acidity and best quality')
    ax1[0].set_title('2.1. Fixed acidity')
    ax1[0].legend()
    plt.hist([dataBestQuality['fixed acidity'],dataWorstQuality['fixed acidity']], color = colors, label = labels)
    plt.legend()


    fig2,ax2 = plt.subplots(2)
    
    ax2[0].plot(dataWorstQuality['volatile acidity'],'+r',label='volatile acidity and worst quality')
    ax2[0].plot(dataBestQuality['volatile acidity'],'+b',label='volatile acidity and best quality')
    ax2[0].set_title('2.1. Volatile acidity')
    ax2[0].legend()
    plt.hist([dataBestQuality['volatile acidity'],dataWorstQuality['volatile acidity']], color = colors, label = labels)
    plt.legend()

    
    fig3,ax3 = plt.subplots(2)
    
    ax3[0].plot(dataWorstQuality['citric acid'],'+r',label='citric acid and worst quality')
    ax3[0].plot(dataBestQuality['citric acid'],'+b',label='citric acid and best quality')
    ax3[0].set_title('2.1. Citric acid')
    ax3[0].legend()
    plt.hist([dataBestQuality['citric acid'],dataWorstQuality['citric acid']], color = colors, label = labels)
    plt.legend()
    
    
    fig4,ax4 = plt.subplots(2)
    
    ax4[0].plot(dataWorstQuality['residual sugar'],'+r',label='residual sugar and worst quality')
    ax4[0].plot(dataBestQuality['residual sugar'],'+b',label='residual sugar and best quality')
    ax4[0].set_title('2.1. Residual sugar')
    ax4[0].legend()
    plt.hist([dataBestQuality['residual sugar'],dataWorstQuality['residual sugar']], color = colors, label = labels)
    plt.legend()
    
    
    fig5,ax5 = plt.subplots(2)
    
    ax5[0].plot(dataWorstQuality['chlorides'],'+r',label='chlorides and worst quality')
    ax5[0].plot(dataBestQuality['chlorides'],'+b',label='chlorides and best quality')
    ax5[0].set_title('2.1. Chlorides')
    ax5[0].legend()
    plt.hist([dataBestQuality['chlorides'],dataWorstQuality['chlorides']], color = colors, label = labels)
    plt.legend()
    
    
    fig6,ax6 = plt.subplots(2)
    
    ax6[0].plot(dataWorstQuality['free sulfur dioxide'],'+r',label='free sulfur dioxide and worst quality')
    ax6[0].plot(dataBestQuality['free sulfur dioxide'],'+b',label='free sulfur dioxide and best quality')
    ax6[0].set_title('2.1. Free sulfur dioxide')
    ax6[0].legend()
    plt.hist([dataBestQuality['free sulfur dioxide'],dataWorstQuality['free sulfur dioxide']], color = colors, label = labels)
    plt.legend()
    
    
    fig7,ax7 = plt.subplots(2)
    
    ax7[0].plot(dataWorstQuality['total sulfur dioxide'],'+r',label='total sulfur dioxide and worst quality')
    ax7[0].plot(dataBestQuality['total sulfur dioxide'],'+b',label='total sulfur dioxide and best quality')
    ax7[0].set_title('2.1. Total sulfur dioxide')
    ax7[0].legend()
    plt.hist([dataBestQuality['total sulfur dioxide'],dataWorstQuality['total sulfur dioxide']], color = colors, label = labels)
    plt.legend()
    
    
    fig8,ax8 = plt.subplots(2)
    
    ax8[0].plot(dataWorstQuality['density'],'+r',label='density and worst quality')
    ax8[0].plot(dataBestQuality['density'],'+b',label='density and best quality')
    ax8[0].set_title('2.1. Density')
    ax8[0].legend()
    plt.hist([dataBestQuality['density'],dataWorstQuality['density']], color = colors, label = labels)
    plt.legend()
    
    
    fig9,ax9 = plt.subplots(2)
    
    ax9[0].plot(dataWorstQuality['pH'],'+r',label='pH and worst quality')
    ax9[0].plot(dataBestQuality['pH'],'+b',label='pH and best quality')
    ax9[0].set_title('2.1. pH')
    ax9[0].legend()
    plt.hist([dataBestQuality['pH'],dataWorstQuality['pH']], color = colors, label = labels)
    plt.legend()
    
    
    fig10,ax10 = plt.subplots(2)
    
    ax10[0].plot(dataWorstQuality['sulphates'],'+r',label='sulphates and worst quality')
    ax10[0].plot(dataBestQuality['sulphates'],'+b',label='sulphates and best quality')
    ax10[0].set_title('2.1. Sulphates')
    ax10[0].legend()
    plt.hist([dataBestQuality['sulphates'],dataWorstQuality['sulphates']], color = colors, label = labels)
    plt.legend()
    
    
    fig11,ax11 = plt.subplots(2)
    
    ax11[0].plot(dataWorstQuality['alcohol'],'+r',label='alcohol and worst quality')
    ax11[0].plot(dataBestQuality['alcohol'],'+b',label='alcohol and best quality')
    ax11[0].set_title('2.1. Alcohol')
    ax11[0].legend()
    plt.hist([dataBestQuality['alcohol'],dataWorstQuality['alcohol']], color = colors, label = labels)
    plt.legend()

