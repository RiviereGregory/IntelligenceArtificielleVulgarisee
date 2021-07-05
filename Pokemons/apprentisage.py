# ***********************
# IMPORT DES MODULES
# ***********************

import pandas as pnd
from sklearn.model_selection import train_test_split

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
