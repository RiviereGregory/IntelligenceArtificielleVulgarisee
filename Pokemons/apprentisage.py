# ***********************
# IMPORT DES MODULES
# ***********************

import joblib
import pandas as pnd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# ***********************
# APPRENTISSAGE
# ***********************

# Chargement des données des Pokemons
dataset = pnd.read_csv("datas/dataset.csv", delimiter='\t')

# Suppresion des données manquante
dataset = dataset.dropna(axis=0, how='any')

# X = on prend toutes les données, mais uniquement les colonnes 4 à 11
#    POINTS_ATTAQUE;POINTS_DEFFENCE;POINTS_ATTAQUE_SPECIALE;POINT_DEFENSE_SPECIALE;POINTS_VITESSE;NOMBRE_GENERATIONS
X = dataset.iloc[:, 5:12].values

# y = on prend uniquement la colonne POURCENTAGE_DE_VICTOIRE (16 ème valeur)
y = dataset.iloc[:, 17].values

# Construction du jeu d'entrainement et du jeu de test
X_APPRENTISSAGE, X_VALIDATION, Y_APPRENTISSAGE, Y_VALIDATION = train_test_split(X, y, test_size=0.2, random_state=0)

# ***************************************
# ALGORITHME 1: REGRESSION LINEAIRE    *
# *************************************

# Choix de l'algorithme
algorithme = LinearRegression()

# Apprentissage à l'aide de la fonction fit
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

# Realisation de la prédiction sur le jeu  de test
predictions = algorithme.predict(X_VALIDATION)

# Calcul de la précision de l'apprentissage à l'aide de la
# fonction r2_score
precision = r2_score(Y_VALIDATION, predictions)

print(">> ----------- REGRESSION LINEAIRE -----------")
print(">> Precision = " + str(precision))
print("------------------------------------------")

#  REGRESSION LINEAIRE
# >> Precision = 0.9043488485570965

# ************************************************************
# ALGORITHME 2: ARBRE DE DECISION APPLIQUE A LA REGRESSION  *
# **********************************************************

# Choix de l'algorithme
algorithme = DecisionTreeRegressor()

# Apprentissage à l'aide de la fonction fit
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

# Realisation de la prédiction sur le jeu  de test
predictions = algorithme.predict(X_VALIDATION)

# Calcul de la précision de l'apprentissage à l'aide de la
# fonction r2_score
precision = r2_score(Y_VALIDATION, predictions)

print(">> ----------- ARBRES DE DECISION -----------")
print(">> Precision = " + str(precision))
print("------------------------------------------")

# ARBRES DE DECISION
# >> Precision = 0.8684246994972593

# *********************************************************
# ALGORITHME 3: RANDOM FOREST APPLIQUE A LA REGRESSION  *
# ******************************************************

# Choix de l'algorithme
algorithme = RandomForestRegressor()

# Apprentissage à l'aide de la fonction fit
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

# Realisation de la prédiction sur le jeu  de test
predictions = algorithme.predict(X_VALIDATION)

# Vérification du score d'apprentissage pour éviter le surapprentissage
precision_apprentissage = algorithme.score(X_APPRENTISSAGE, Y_APPRENTISSAGE)
print(">> precision_apprentissage = " + str(precision_apprentissage))
# Calcul de la précision de l'apprentissage à l'aide de la
# fonction r2_score
precision = r2_score(Y_VALIDATION, predictions)

print(">> ----------- RANDOM FOREST -----------")
print(">> Precision = " + str(precision))
print("------------------------------------------")

# RANDOM FOREST
# >> Precision = 0.9376935860217895


# Résultat ici c'est le random forest qui a le meilleur score

# Sauvegarde de l'algorithme
fichier = 'modele/modele_pokemon.mod'
joblib.dump(algorithme, fichier)
