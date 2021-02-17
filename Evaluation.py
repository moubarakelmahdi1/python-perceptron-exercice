# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:22:24 2020

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

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn import tree

from sklearn.model_selection import cross_val_score

def evaluationClassifieurs(X, y, scores, accuracy):
    """
    Cette fonction permet d'évaluer les classifieurs
    avec la méthode de la validation croisée
    
    Elle retourne un dataframe avec les scores moyens 
    et une liste avec tous les scores accuracy obtenus 
    lors de la validation croisée sur chaque classifieur
    """
    
    
    lrClassifier  = LogisticRegression() 
    print('*** Scores moyens pour la régression logistique :')
    accuracy.append(cross_val_score(lrClassifier, X, y, cv=10, scoring = 'accuracy'))
    scores['Accuracy']['lr']=accuracy[0].mean()
    scores['Precision']['lr']=cross_val_score(lrClassifier, X, y, cv=10, scoring = 'precision').mean()
    scores['Recall']['lr']=cross_val_score(lrClassifier, X, y, cv=10, scoring = 'recall').mean()
    print('    - accuracy : ', scores['Accuracy']['lr'])
    print('    - precision : ',scores['Precision']['lr'])
    print('    - recall : ',scores['Recall']['lr'])
    
    print('\n*** Scores moyens pour la machine à vecteur support :')
    svmClassifier = svm.SVC(kernel='linear')
    accuracy.append(cross_val_score(svmClassifier, X, y, cv=10, scoring = 'accuracy'))
    scores['Accuracy']['svm']=accuracy[1].mean()
    scores['Precision']['svm']=cross_val_score(svmClassifier, X, y, cv=10, scoring = 'precision').mean()
    scores['Recall']['svm']=cross_val_score(svmClassifier, X, y, cv=10, scoring = 'recall').mean()
    print('    - accuracy : ', scores['Accuracy']['svm'])
    print('    - precision : ',scores['Precision']['svm'])
    print('    - recall : ',scores['Recall']['svm'])
    
    print("\n*** Scores moyens pour l'analyse discriminante linéaire :")
    ldaClassifier = LinearDiscriminantAnalysis()
    accuracy.append(cross_val_score(ldaClassifier, X, y, cv=10, scoring = 'accuracy'))
    scores['Accuracy']['lda']=accuracy[2].mean()
    scores['Precision']['lda']=cross_val_score(ldaClassifier, X, y, cv=10, scoring = 'precision').mean()
    scores['Recall']['lda']=cross_val_score(ldaClassifier, X, y, cv=10, scoring = 'recall').mean()
    print('    - accuracy : ', scores['Accuracy']['lda'])
    print('    - precision : ',scores['Precision']['lda'])
    print('    - recall : ',scores['Recall']['lda'])
    
    
    print("\n*** Scores moyens pour l'analyse discriminante quadratique :")
    qdaClassifier = QuadraticDiscriminantAnalysis()
    accuracy.append(cross_val_score(qdaClassifier, X, y, cv=10, scoring = 'accuracy'))
    scores['Accuracy']['qda']=accuracy[3].mean()
    scores['Precision']['qda']=cross_val_score(qdaClassifier, X, y, cv=10, scoring = 'precision').mean()
    scores['Recall']['qda']=cross_val_score(qdaClassifier, X, y, cv=10, scoring = 'recall').mean()
    print('    - accuracy : ', scores['Accuracy']['qda'])
    print('    - precision : ',scores['Precision']['qda'])
    print('    - recall : ',scores['Recall']['qda'])
    
    print("\n* Choix des meilleurs paramètres pour les k-plus proches voisins :\n")
    
    kneighborsClassifier = KNeighborsClassifier()
    kneighborsClassifier.fit(X,y)
    k = np.arange(1, 15)
    
    train_score, val_score = validation_curve(kneighborsClassifier, X, y,'n_neighbors', k, cv=10)
    fig1,ax1 = plt.subplots()
    plt.plot(k, val_score.mean(axis=1), label='validation score')
    plt.plot(k, train_score.mean(axis=1), label='training score')
    plt.ylabel('score')
    plt.xlabel('n_neighbors')
    plt.legend()
    print("-> Graphique représentant l'évolution des scores de validation et d'entrainement en fonction du nombre de voisins.")
    
    param_grid = {'n_neighbors': np.arange(1, 15),'metric': ['euclidean', 'manhattan']}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)
    grid.fit(X, y)
    print("-> Meilleur score : ")
    print(grid.best_score_)
    print("-> Meilleurs paramètres : ")
    print(grid.best_params_)
    kneighborsClassifier = grid.best_estimator_
    print("\n--> Scores du meilleur estimateur : ")
    accuracy.append(cross_val_score(kneighborsClassifier, X, y, cv=10, scoring = 'accuracy'))
    scores['Accuracy']['Kneighbors']=accuracy[4].mean()
    scores['Precision']['Kneighbors']=cross_val_score(kneighborsClassifier, X, y, cv=10, scoring = 'precision').mean()
    scores['Recall']['Kneighbors']=cross_val_score(kneighborsClassifier, X, y, cv=10, scoring = 'recall').mean()
    print('    - accuracy : ', scores['Accuracy']['Kneighbors'])
    print('    - precision : ',scores['Precision']['Kneighbors'])
    print('    - recall : ',scores['Recall']['Kneighbors'])
    
    
    print("\n*** Scores moyens pour les arbres de décision :")
    treeClassifier = tree.DecisionTreeClassifier(random_state=5)
    accuracy.append(cross_val_score(treeClassifier, X, y, cv=10, scoring = 'accuracy'))
    scores['Accuracy']['tree']=accuracy[5].mean()
    scores['Precision']['tree']=cross_val_score(treeClassifier, X, y, cv=10, scoring = 'precision').mean()
    scores['Recall']['tree']=cross_val_score(treeClassifier, X, y, cv=10, scoring = 'recall').mean()
    print('    - accuracy : ', scores['Accuracy']['tree'])
    print('    - precision : ',scores['Precision']['tree'])
    print('    - recall : ',scores['Recall']['tree'])
    
    print('\n* Tableau reprenant tous les scores moyens pour chaque classifieur :\n',scores)
    
    print("-> Voir le graphique Accuracy scores pour les boîtes à moustache des scores accuracy pour chaque classifieur.")
    fig2 = plt.subplots()
    boxplotElements = plt.boxplot(accuracy)
    plt.gca().xaxis.set_ticklabels(['lr', 'svm', 'lda', 'qda', 'kneighbors', 'tree'])
    plt.ylim(0.5, 0.9)
    plt.title('Accuracy scores')
        
    return scores, accuracy