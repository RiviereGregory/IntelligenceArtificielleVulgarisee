#-----------------------------------------------------------------------------------------
# @Auteur : Aurélien Vannieuwenhuyze
# @Entreprise : Junior Makers Place
# @Livre
# @Chapitre : 06 - Machine Learning et Pokemon deuxième
#
# Modules necessaires :
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#   MATPLOTLIB 3.0.3
#   SEABORN 0.9.0
#   SCIKIT-LEARN 0.20.3
#
# Pour installer un module :
#   Cliquer sur le menu File > Settings > Project:nom_du_projet > Project interpreter > bouton +
#   Dans la zone de recherche en haut à gauche saisir le nom du module
#   Choisir la version en bas à droite
#   Cliquer sur le bouton install situé en bas à gauche
#-----------------------------------------------------------------------------------------

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Utilisation du module Pandas
import pandas as pnd

#Chargement du dataset
dataset = pnd.read_csv("datas/dataset.csv",delimiter='\t')

#Suppression des valeur NA (colonnes : Premier Pokemon, Second Pokemon)
dataset = dataset.dropna(axis=0, how='any')


#X = on prend toutes les données, mais uniquement les colonnes 4 à 11
#    POINTS_ATTAQUE;POINTS_DEFFENCE;POINTS_ATTAQUE_SPECIALE;POINT_DEFENSE_SPECIALE;POINTS_VITESSE;NOMBRE_GENERATIONS
X = dataset.iloc[:, 5:12].values

#y = on prend uniquement la colonne POURCENTAGE_DE_VICTOIRE (16 ème valeur)
y = dataset.iloc[:, 17].values


#Construction du jeu d'entrainement et du jeu de test
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_APPRENTISSAGE, X_VALIDATION, Y_APPRENTISSAGE, Y_VALIDATION = train_test_split(X, y, test_size = 0.2, random_state = 0)





#---- ALGORITHME 1: REGRESSION LINEAIRE -----
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression

#Choix de l'algorithme
algorithme = LinearRegression()

#Apprentissage à l'aide de la fonction fit
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

#Realisation de la prédiction sur le jeu  de test
predictions = algorithme.predict(X_VALIDATION)

#Calcul de la précision de l'apprentissage à l'aide de la
#fonction r2_score
precision = r2_score(Y_VALIDATION, predictions)


print(">> ----------- REGRESSION LINEAIRE -----------")
print(">> Precision = "+str(precision))
print("------------------------------------------")



#---- ALGORITHME 2: ARBRE DE DECISION APPLIQUE A LA REGRESSION-----


#Choix de l'algorithme
from sklearn.tree import DecisionTreeRegressor
algorithme = DecisionTreeRegressor()

#Apprentissage à l'aide de la fonction fit
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

#Realisation de la prédiction sur le jeu  de test
predictions = algorithme.predict(X_VALIDATION)

#Calcul de la précision de l'apprentissage à l'aide de la
#fonction r2_score
precision = r2_score(Y_VALIDATION, predictions)


print(">> ----------- ARBRES DE DECISION -----------")
print(">> Precision = "+str(precision))
print("------------------------------------------")




#Choix de l'algorithme
from sklearn.ensemble import RandomForestRegressor
algorithme = RandomForestRegressor()

#Apprentissage à l'aide de la fonction fit
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

#Realisation de la prédiction sur le jeu  de test
predictions = algorithme.predict(X_VALIDATION)

#Calcul de la précision de l'apprentissage à l'aide de la
#fonction r2_score
precision = r2_score(Y_VALIDATION, predictions)


print(">> ----------- FORETS ALEATOIRES -----------")
print(">> Precision = "+str(precision))
print("------------------------------------------")


#Sauvegarde de l'algorithme
from sklearn.externals import joblib
fichier = 'modele/modele_pokemon.mod'
joblib.dump(algorithme, fichier)