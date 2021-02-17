# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:06:20 2020

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
import seaborn as sns
from statistics import mean

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from Analyse import *
from Preparation import *
from Evaluation import *
from Perceptron import *

pd.set_option('display.max_columns', None)

data = pd.read_csv('red_wines.csv', engine = 'python')

print('\n_________DEBUT DU PROJET_________\n\n')


print('\n~ 2.1. ANALYSE DU PROBLEME ~')

"""
Toutes les fonctions utilisées dans cette partie
sont retrouvables dans le fichier Analyse.py
"""


print('\n*** Types des attributs dans le dataframe : ')
print(data.dtypes)

print("\n*** Suite à l'observation des données, nous déterminons qu'il faudra prédire la qualité (quality).\nLes valeurs prises par cet attribut sont : ")
print(allQualityValues(data))
print('Nous avons deux valeurs possibles donc deux classes (qualité = -1 ou qualité = +1).\nNous allons donc opter pour un classifieur binaire\n')

print("*** Observons la répartition des qualités en fonction de chaque attribut afin d'en savoir plus sur nos données")
print("Les 12 graphiques apparaissent dans la section appropriée")
graphiquesRepartitionClasses(data)



print('\n\n~ 2.2. PREPARATION DES DONNEES ~')

"""
Toutes les fonctions utilisées dans cette partie
sont retrouvables dans le fichier Preparation.py
"""

print('\n*** Cherchons si des valeurs sont manquantes dans notre dataframe : ')
print("Nombre de valeurs manquantes trouvées :", len(ValuesMissing(data)))
data = replaceValuesMissing(data)
print("Nombre de valeurs manquantes restantes après remplacement :", len(ValuesMissing(data)))

        
print('\n*** Cherchons si des valeurs sont non numériques : ')
print(ValuesNotNumbers(data))
print('Les seules valeurs non numériques sont les titres des colonnes.')

print('\n*** Recherchons à présent les valeurs aberrantes :')

print('\n     - Valeurs aberrantes pour pH :')
print('\nVoici les pH maximaux (supérieurs à 4):')
print(data[data['pH']>4]['pH'])
    
print("\nNous allons à présent détecter les données aberrantes avec le critère des trois sigmas sur les attributs qui semblent suivre une loi normale d'après leur distribution. Nous remplaçons ensuite ces données aberrantes par la moyenne de l'attribut concerné.\n")


replacedData = replaceAbnormalValues(data)
print('\nValeurs des pH supérieurs à 4 dans le dataframe avec les valeurs aberrantes remplacées')
print(replacedData[replacedData['pH']>4]['pH'])
print(replacedData.shape)
print()


      
print('\n*** Observons à présent la distribution des données pour chaque attribut sur les 4 figures regroupant les distributions 3 par 3')
graphiquesDistributionsAttributs(replacedData)


print("\n*** Cherchons désormais à déterminer s'il existe des corrélations entre nos attributs")
print("\nObservons les coefficients de corrélation des attributs 2 à 2 : ")
correlation = replacedData.corr()
print("Voici le tableau des coefficients de corrélation :", correlation)

print("\nVoici un graphique reprenant ce tableau des coefficients de corrélation (intitulé * Correlation coefficients *)")
figE, axE = plt.subplots()
axE=sns.heatmap(correlation, xticklabels=correlation.columns.values, yticklabels=correlation.columns.values)
axE.set(title = "* Correlation coefficients *")

print("\nOn identifie grâce à ce graphique que certains attributs ont une corrélation significative :")
print('    - fixed acidity & citric acid -> ',correlation['fixed acidity']['citric acid'])
print('    - volatile acidity & citric acid -> ',correlation['volatile acidity']['citric acid'])
print('    - fixed acidity & density -> ',correlation['fixed acidity']['density'])
print('    - fixed acidity & pH -> ',correlation['fixed acidity']['pH'])

print("\nVoici une autre figure représentant les distributions jointes des attributs 2 à 2")
axCorr = sns.pairplot(replacedData)


print("\n -> Nous décidons de supprimer les attributs density, pH et citric acid")
replacedData = replacedData.drop(['density', 'pH', 'citric acid'], axis=1)



print("\n*** Calculons la proportion des données dans chaque classe :")
print("      - Proportion des données dans la classe bonne qualité :", replacedData[replacedData['quality'] == 1].shape[0])
print("      - Proportion des données dans la classe mauvaise qualité :", replacedData[replacedData['quality'] == -1].shape[0])



print("\n*** Nous allons à présent centrer et réduire les attributs qu'il nous reste")
replacedData = centrer_reduire(replacedData)
print("Moyennes :\n", replacedData.mean(axis = 0))
print("-> Les moyennes sont bien environ égales à 0\n")

print("Variances :\n", replacedData.std(axis = 0))
print("-> Les variances sont bien environ égales à 1\n")

print("\n*** Nous décidons également de normaliser nos données entre 0 et 1")
y = replacedData['quality']
X = replacedData.drop(['quality'],axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)


print('\n\n~ 2.3. DEFINITION DU PROTOCOLE EXPERIMENTAL ~\n')

print("\n*** Pour chaque classifieur, nous allons appliquer la méthode de la validation croisée.")
print("\n* Les métriques que nous décidons d'évaluer sont :")
print("\n     - Le taux de reconnaissance (accuracy)")
print("\n     - La précision (precision)")
print("\n     - Le rappel (recall)")


scores = pd.DataFrame(index = ['lr', 'svm', 'lda','qda','Kneighbors','tree'], columns = ['Accuracy', 'Precision', 'Recall'], dtype='float')


print('\n\n~ 2.4. CHOIX, ENTRAINEMENT ET EVALUATION DU CLASSIFIEUR ~\n')

"""
Toutes les fonctions utilisées dans cette partie
sont retrouvables dans le fichier Evaluation.py
"""

accuracy = []
scores, accuracy = evaluationClassifieurs(X, y, scores, accuracy)


print('\n\n~ 2.5. ANALYSE DES RESULTATS ~\n')

print("\n*** Pour nos données manquantes et aberrantes, nous avions fait le choix de les remplacer par la moyenne correspondante.")
print("\n* Afin de déterminer si ce choix était correct, nous allons effectuer à nouveau le 2.4. en utilisant une autre méthode pour la gestion des données manquantes et aberrantes : les supprimer.")
print("\n* De plus, nous n'allons pas centrer et réduire nos données et nous n'allons pas les normaliser.")

##Faire removedData
removedData = deleteAbnormalValues(data)
removedData.dropna(inplace=True)

y2 = removedData['quality']
X2 = removedData.drop(['quality'],axis=1)

print("\n* Afin d'observer les résultats que nous avons, merci de décommenter la ligne 181.")

scores2 = pd.DataFrame(index = ['lr', 'svm', 'lda','qda','Kneighbors','tree'], columns = ['Accuracy', 'Precision', 'Recall'], dtype='float')
accuracy2 = []
#scores2, accuracy2 = evaluationClassifieurs(X2, y2, scores2, accuracy2)
print("\n-> Nous observons alors que pour tous les classifieurs nous avons un warning nous indiquant que les classifieurs ne sont pas parvenus à converger. Ce résultat est certainement du au fait que nous n'avons pas centré et réduit nos données.")

print("\n* Faisons le même test mais en appliquant la fonction centrer et réduire à nos données cette fois. Afin d'observer les résultats que nous avons, merci de décommenter la ligne 187. ")

X2 = centrer_reduire(X2)
#scores2, accuracy2 = evaluationClassifieurs(X2, y2, scores2, accuracy2)

print("\n* Faisons le même test mais en appliquant la fonction centrer et réduire à nos données et en les normalisant cette fois. Voici les résultats que nous avons donc en modifiant uniquement notre manière de gérer les données manquantes et aberrantes :\n")
X2 = scaler.fit_transform(X2)
scores2, accuracy2 = evaluationClassifieurs(X2, y2, scores2, accuracy2)

print("\n---> Ces différentes comparaisons nous montrent que nous n'avons pas choisi la bonne méthode de gestion des données aberrantes et manquantes. Nous décidons donc de poursuivre le projet avec ces données aberrantes et manquantes supprimées de notre dataframe. Nous prendrons donc en compte les résultats juste au-dessus et pas les premiers que nous avons obtenu. ")
#Nous poursuivons donc le projet avec X2 et y2

print("\n\n~ 3. IMPLEMENTATION DE L'ALGORITHME DE PERCEPTRON ~\n")

"""
Toutes les fonctions utilisées dans cette partie
sont retrouvables dans le fichier Perceptron.py
"""

testPerceptron()

#On divise nos données en 3 jeux, un jeu d'entrainement, un jeu de test et un jeu de validation
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.1,random_state=5)

X_val = X_train[0:719]
y_val = y_train[0:719]

X_train = X_train[720:1439]
y_train = y_train[720:1439]


#Le jeu de validation est utilisé pour optimiser
#les méta-paramètres de notre perceptron
perceptronClassifier = optimisePerceptron(X_val, y_val)

print("\n*** Nous entrainons à présent notre perceptron optimisé sur les données d'entrainement")
perceptronClassifier.fit(X_train, y_train)
print("\n* Voici le score de notre perceptron optimisé sur les données de validation : ", perceptronClassifier.score(X_val, y_val))
print("\n* Voici le score de notre perceptron optimisé mais sur les données de test cette fois : ", perceptronClassifier.score(X_test, y_test))

print("\n*** Il nous reste à évaluer le perceptron au même titre que les autres classifieurs. C'est-à-dire par la méthode de la validation croisée et en évaluant les mêmes métriques.")
accuracy, precision, recall = perceptronClassifier.cross_validation(X2, y2)
print("\n*** Scores moyens pour le perceptron optimisé :")
print('    - accuracy : ', mean(accuracy))
print('    - precision : ', mean(precision))
print('    - recall : ', mean(recall))

print("-> Voir le graphique Accuracy scores perceptron pour la boîte à moustache des scores accuracyde notre perceptron optimisé.")
fig = plt.subplots()
boxplotElements = plt.boxplot(accuracy)
plt.gca().xaxis.set_ticklabels(['Perceptron'])
plt.ylim(0.5, 0.9)
plt.title('Accuracy scores perceptron')

print('\n_________FIN DU PROJET_________\n\n')
