# ***********************
# IMPORT DES MODULES
# ***********************
import os

import matplotlib.pyplot as plt
import pandas as pnd
import seaborn as sns

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

# Statistiques
print(nouveauPokedex.describe())

# Visualisation des pokemons de type 1
axe_X = sns.countplot(x="TYPE_1", hue="LEGENDAIRE", data=nouveauPokedex)
plt.xticks(rotation=90)
plt.xlabel('TYPE_1')
plt.ylabel('Total')
plt.title("POKEMONS DE TYPE_1")
plt.show()

# Visualisation des pokemons de type 2
axe_X = sns.countplot(x="TYPE_2", hue="LEGENDAIRE", data=nouveauPokedex)
plt.xticks(rotation=90)
plt.xlabel('TYPE_2')
plt.ylabel('Total')
plt.title("POKEMONS DE TYPE_2")
plt.show()

# ***********************************************
# Résultat les pokémons dont l'histogramme est le plus important
# Premier type Herbe, Eau, Insecte et Normal
# Second type Vol, Poison et Sol
# ***********************************************

print(nouveauPokedex.groupby('TYPE_1').agg({"POURCENTAGE_DE_VICTOIRES": "mean"}).sort_values(
    by="POURCENTAGE_DE_VICTOIRES"))

# Résultat
# Fée                         0.329300
# Roche                       0.404852
# Acier                       0.424529
# Poison                      0.433262
# Insecte                     0.439006
# Glace                       0.439604
# Herbe                       0.440364
# Eau                         0.469357
# Combat                      0.475616
# Spectre                     0.484027
# Normal                      0.535578
# Sol                         0.541526
# Psy                         0.545747
# Feu                         0.579215
# Obscur                      0.629726
# Electrique                  0.632861
# Dragon                      0.633587
# Vol                         0.765061

# *********************************
# Recherche correlation
# *********************************
corr = nouveauPokedex.loc[:,
       ['TYPE_1', 'POINTS_DE_VIE', 'POINTS_ATTAQUE', 'POINTS_DEFFENCE', 'POINTS_ATTAQUE_SPECIALE',
        'POINT_DEFENSE_SPECIALE', 'POINTS_VITESSE', 'LEGENDAIRE', 'POURCENTAGE_DE_VICTOIRES']].corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
plt.show()

# Résultat :
# POINTS_VITESSE correlation de 0.94 avec POURCENTAGE_DE_VICTOIRES
# POINTS_ATTAQUE correlation de 0.5 avec POURCENTAGE_DE_VICTOIRES
# Par contre deception pour
# LEGENDAIRE correlation de 0.33 avec POURCENTAGE_DE_VICTOIRES
