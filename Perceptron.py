# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:47:18 2020

@author: estel
"""

"""
Projet IDS 
Duhem Estelle - Moubarak El Mahdi 
INGE1 (L3) - Fi22
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn import datasets

from sklearn.model_selection import train_test_split



class Perceptron:

    def __init__(self, lr=0.01, epochs=100):
        """
        Constructeur de notre classe
        """
        
        self.lr = lr
        self.epochs = epochs



    def fit(self, X, Y):
        
        X = np.array(X)
        Y = np.array(Y)

        self.weight = np.ones(X.shape[1] + 1)

        n_iter = 0
        error = 1
        while(n_iter <= self.epochs and error > 0.1):
            for x, y in zip(X, Y):
                predict = self.predict_unit(x)

                if predict != y:
                    self.weight[1:] += self.lr * y * x
                    self.weight[0] += self.lr * y
            error = 1 - self.score(X, Y)
            n_iter +=1
        return self


    def predict_unit(self, X):
        """
        Fonction permettant de prédire un échantillon de nos données
        
        Retourne une seule prédiction (-1 ou 1)
        """
        
        X = np.array(X)
        predict =  np.dot(X, self.weight[1:]) + self.weight[0]
        if predict>0:
            return 1
        else:
            return -1

    def predict(self, X):
        """
        Fonction permettant de prédire toutes nos données
        
        Retourne un tableau de prédictions
        """
        
        X = np.array(X)
        predictions = []
        for x in X:
            predict = self.predict_unit(x)
            if predict > 0:
                predictions.append(1)
            else:
                predictions.append(-1)

        return predictions

    
    def score(self, X, Y):
        """
        Fonction qui retourne le score du perceptron sur les données X
        Ce score correspond à accuracy
        """
        
        i=0
        for x,y in zip(self.predict(X),Y):
             if x==y:
                 i = i+1
             elif x==-1 and y==0:
                 i = i+1
        return i/len(X) 
    
    def cross_validation(self, X, Y):
        """
        Fonction qui permet de faire la validation croisée
        sur le perceptron 
        
        Retourne 3 listes avec tous les scores obtenus
        pour accuracy, recall et precision
        """
        
        accuracy = []
        recall = []
        precision = []
        
        X = np.array(X)
        Y = np.array(Y)
            
        kf = KFold(n_splits = 10)
        for train_index , test_index in kf.split(X):
            x_train, x_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
            self.fit(x_train, y_train)
            prediction = self.predict(x_test)
            accuracy.append(accuracy_score(y_test, prediction))
            recall.append(recall_score(y_test, prediction))
            precision.append(precision_score(y_test, prediction))
            
        return accuracy, recall, precision

def testPerceptron():
    """
    Cette fonction permet de tester notre implémentation 
    de l'algorithme du perceptron sur des données 
    linéairement séparables
    """
    
    print("\n*** Test du modèle Perceptron sur des données linéairement séparables : ")
    
    A, B = datasets.make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 10))
    A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.1,random_state=5)
    
    fig1,ax1 = plt.subplots()
    plt.plot(A[:, 0][B == 0], A[:, 1][B == 0], 'g^')
    plt.plot(A[:, 0][B == 1], A[:, 1][B == 1], 'bs')
    plt.show()
    modeleTest = Perceptron()
    modeleTest.fit(A_train,B_train)
    print("Score (correspondant à accuracy) : ", modeleTest.score(A_test,B_test))

def optimisePerceptron(X_val, y_val):
    """
    Cette fonction permet d'optimiser le perceptron sur les données
    entrées en paramètres
    
    Retourne un perceptron avec le meilleur taux d'apprentissage
    """
    
    print("\n*** Réalisons à présent l'optimisation du taux d'apprentissage (learning rate) de notre perceptron sur des données de validation.")

    score = []
    bestLr = 0
    for j in np.arange(1,0.01, -0.01):
        modele2= Perceptron(lr=j)
        modele2.fit(X_val,y_val)
        temp=modele2.score(X_val,y_val)
        if j!=1 and temp>max(score):
            bestLr = j
        score.append(modele2.score(X_val,y_val))
    fig3,ax3 = plt.subplots()
    plt.plot(np.arange(1,0.01, -0.01),score, label="score en fonction du taux d'apprentissage")
    plt.legend()
    print("\n* Meilleur taux d'apprentissage : ",bestLr)
    print("\n-> Voir le graphique représentant le score en fonction du taux d'apprentissage")


    return Perceptron(lr = bestLr)

