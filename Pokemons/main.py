# ***********************
# IMPORT DES MODULES
# ***********************
import os

import pandas as pnd

# désactivation du nombre maximun de colonnes du dataframe à afficher
pnd.set_option('display.max_columns', None)

# ***********************
# ANALYSE DES DONNEES
# ***********************

# Récupération des fichiers contenus dans datas
listeDeFichiers = os.listdir("datas")

# Nom des fichiers
for fichier in listeDeFichiers:
    print(fichier)

# Chargement des données des Pokemons
nosPokemons = pnd.read_csv("datas/pokedex.csv")

# Affichage des colonnes du dataFrame
print(nosPokemons.columns.values)

# Affichage des 10 premières lignes
print(nosPokemons.head(10))

# Transformation de la colonne LEGENDAIRE en entire 0 = FAUX et 1 = VRAI
nosPokemons['LEGENDAIRE'] = (nosPokemons['LEGENDAIRE'] == 'VRAI').astype(int)
print(nosPokemons['LEGENDAIRE'].head(800))

# Comptage du nombre d'observation et de features
print(nosPokemons.shape)

# Information sur le jeu de données (vérification que les données sont completes)
print(nosPokemons.info())

# recherche et affichage des données manquante
print(nosPokemons[nosPokemons['NOM'].isnull()])

# recherche de la valeur manquante
print(nosPokemons['NOM'][61])
print(nosPokemons['NOM'][63])
# Férosinge et Caninos
# https://www.pokepedia.fr/Liste_des_Pok%C3%A9mon_dans_l%27ordre_du_Pok%C3%A9dex_National
# le nom manquant est Colossinge
nosPokemons['NOM'][62] = 'Colossinge'
print(nosPokemons['NOM'][62])

# Chargement des données des Combats
combats = pnd.read_csv("datas/combats.csv")

# Affichage des colonnes du dataFrame
print(combats.columns.values)

# Affichage des 10 premières lignes
print(combats.head(10))

# Comptage du nombre d'observation et de features
print(combats.shape)

# Information sur le jeu de données (vérification que les données sont completes)
print(combats.info())
