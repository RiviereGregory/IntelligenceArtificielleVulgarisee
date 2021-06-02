#-----------------------------------------------------------------------------------------
# @Auteur : Aurélien Vannieuwenhuyze
# @Entreprise : Junior Makers Place
# @Livre
# @Chapitre : 06 - Machine Learning et Pokemon premiere partie
#
# Modules necessaires :
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#
# Pour installer un module :
#   Cliquer sur le menu File > Settings > Project:nom_du_projet > Project interpreter > bouton +
#   Dans la zone de recherche en haut à gauche saisir le nom du module
#   Choisir la version en bas à droite
#   Cliquer sur le bouton install situé en bas à gauche
#-----------------------------------------------------------------------------------------


#------------------------------------------
# IMPORT DES MODULES
#------------------------------------------
import os #Utilisation du module OS (operating system)


#Utilisation du module Pandas
import pandas as pnd

#Desactivation du nombre maximum de colonnes du DataFrame à afficher
pnd.set_option('display.max_columns',None)


#------------------------------------------
# ANALYSE DES DONNEES
#------------------------------------------

#Recupération des fichiers contenus dans le répertoire datas
#de notre projet
listeDeFichiers = os.listdir("datas")

#Quel est le nom de chaque fichier ?
for fichier in listeDeFichiers:
    print(fichier)


#Chargement des données des Pokemons dans un
#Dataframe nommé nosPokemons
nosPokemons = pnd.read_csv("datas/pokedex.csv")

#Affichage des colonnes du Dataframe
print(nosPokemons.columns.values)

#Affichage des 10 premières lignes du DataFrame
print(nosPokemons.head(10))

#Transformation de la colonne Legendaire en entier 0= FAUX et 1=VRAI
nosPokemons['LEGENDAIRE'] = (nosPokemons['LEGENDAIRE']=='VRAI').astype(int)
print(nosPokemons['LEGENDAIRE'].head(800))


#Comptage du nombre d'observations et de features
print (nosPokemons.shape)

#Informations sur notre jeu de données
print (nosPokemons.info())

#Recherche du Pokemon dont le nom est manquant
print(nosPokemons[nosPokemons['NOM'].isnull()])
print(nosPokemons['NOM'][61])
print(nosPokemons['NOM'][63])
nosPokemons['NOM'][62] = "Colosinge"
print(nosPokemons['NOM'][62])

#Chargement des données des combats
combats = pnd.read_csv("datas/combats.csv")

#Affichage des colonnes du Dataframe
print(combats.columns.values)

#Affichage des 10 premières lignes du Dataframe
print(combats.head(10))

#Comptage du nombre de lignes et de colonnes
print (combats.shape)

#Informations sur notre jeu de données
print (combats.info())


#Agregation des victoires en premiere et seconde position
nbFoisPremierePosition = combats.groupby('Premier_Pokemon').count()
print(nbFoisPremierePosition)

nbFoisSecondePosition = combats.groupby('Second_Pokemon').count()
print(nbFoisSecondePosition)

nombreTotalDeCombats = nbFoisPremierePosition + nbFoisSecondePosition
print(nombreTotalDeCombats)

nombreDeVictoires = combats.groupby('Pokemon_Gagnant').count()
print(nombreDeVictoires)


#On crée une liste à partir d'une extraction pour obtenir la liste des Pokemons que l'on trie par numéro
#Cette liste de numéros qui nous permettra réaliser l'agrégation des données
listeAAgreger = combats.groupby('Pokemon_Gagnant').count()
listeAAgreger.sort_index()

#On ajoute le nombre de combats
listeAAgreger['NBR_COMBATS'] = nbFoisPremierePosition.Pokemon_Gagnant + nbFoisSecondePosition.Pokemon_Gagnant

#On ajoute le nombre de victoires
listeAAgreger['NBR_VICTOIRES'] = nombreDeVictoires.Premier_Pokemon

#On calcule le pourcentage de victoires
listeAAgreger['POURCENTAGE_DE_VICTOIRES']= nombreDeVictoires.Premier_Pokemon/(nbFoisPremierePosition.Pokemon_Gagnant + nbFoisSecondePosition.Pokemon_Gagnant)

#On affiche la nouvelle liste
print(listeAAgreger)

#Création d'un nouveau Pokedex comportant les noms de Pokemons et leur victoire
nouveauPokedex = nosPokemons.merge(listeAAgreger, left_on='NUMERO', right_index = True, how='left')

print(nouveauPokedex)






