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
# encoding="ISO-8859-1" pour lire correctement les accents
nosPokemons = pnd.read_csv("datas/pokedex.csv", encoding="ISO-8859-1")

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

# Nombre de combat fait par pokemon
# Attention les noms des colonnes sont trompeuse
# Second_Pokemon, Premier_Pokemon, Pokemon_Gagnant
# En vérité c'est :
# Numéro du pokemon, nb de fois en second position, nb de fois en second position
nbFoisPremierePosition = combats.groupby('Premier_Pokemon').count()
print(nbFoisPremierePosition)
# Premier_Pokemon, Second_Pokemon, Pokemon_Gagnant
# En vérité c'est :
# Numéro du pokemon, nb de fois en première position, nb de fois en première position
nbFoisSecondePosition = combats.groupby('Second_Pokemon').count()
print(nbFoisSecondePosition)
# Agregation des 2 tableau
nombreTotalDeCombats = nbFoisPremierePosition + nbFoisSecondePosition
print(nombreTotalDeCombats)
nombreDeVictoires = combats.groupby('Pokemon_Gagnant').count()
print(nombreDeVictoires)

# On crée une liste à partir d'une extraction pour obtenir la liste des Pokemons que l'on trie par numéro
# Cette liste de numéros qui nous permettra réaliser l'agrégation des données
listeAAgreger = combats.groupby('Pokemon_Gagnant').count()
listeAAgreger.sort_index()

# On ajoute le nombre de combats
listeAAgreger['NBR_COMBATS'] = nbFoisPremierePosition.Pokemon_Gagnant + nbFoisSecondePosition.Pokemon_Gagnant

# On ajoute le nombre de victoires
listeAAgreger['NBR_VICTOIRES'] = nombreDeVictoires.Premier_Pokemon

# On calcule le pourcentage de victoires
listeAAgreger['POURCENTAGE_DE_VICTOIRES'] = nombreDeVictoires.Premier_Pokemon / (
        nbFoisPremierePosition.Pokemon_Gagnant + nbFoisSecondePosition.Pokemon_Gagnant)

# On affiche la nouvelle liste
print(listeAAgreger)

# Agregation avec le pokédex
nouveauPokedex = nosPokemons.merge(listeAAgreger, left_on='NUMERO', right_index=True, how='left')
print(nouveauPokedex)
nouveauPokedex.to_csv('datas/nouveau_pokedex.csv', encoding='utf-8', index=False)
