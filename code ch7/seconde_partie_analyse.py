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

#Utilisation du module Pandas
import pandas as pnd

#Desactivation du nombre maximum de colonnes du DataFrame à afficher
pnd.set_option('display.max_columns',None)


#------------------------------------------
# ANALYSE DES DONNEES
#------------------------------------------



#Reprise du code de la première partie
nosPokemons = pnd.read_csv("datas/pokedex.csv")
nosPokemons['LEGENDAIRE'] = (nosPokemons['LEGENDAIRE']=='VRAI').astype(int)
#nosPokemons['NOM'][62] = "Colosinge"
combats = pnd.read_csv("datas/combats.csv")
nbFoisPremierePosition = combats.groupby('Premier_Pokemon').count()
nbFoisSecondePosition = combats.groupby('Second_Pokemon').count()
nombreTotalDeCombats = nbFoisPremierePosition + nbFoisSecondePosition
nombreDeVictoires = combats.groupby('Pokemon_Gagnant').count()
listeAAgreger = combats.groupby('Pokemon_Gagnant').count()
listeAAgreger.sort_index()
listeAAgreger['NBR_COMBATS'] = nbFoisPremierePosition.Pokemon_Gagnant + nbFoisSecondePosition.Pokemon_Gagnant
listeAAgreger['NBR_VICTOIRES'] = nombreDeVictoires.Premier_Pokemon
listeAAgreger['POURCENTAGE_DE_VICTOIRES']= nombreDeVictoires.Premier_Pokemon/(nbFoisPremierePosition.Pokemon_Gagnant + nbFoisSecondePosition.Pokemon_Gagnant)
nouveauPokedex = nosPokemons.merge(listeAAgreger, left_on='NUMERO', right_index = True, how='left')

#Seconde partie

import matplotlib.pyplot as plt
import seaborn as sns

#Visualisation des Pokemons de type 1
axe_X = sns.countplot(x="TYPE_1", hue="LEGENDAIRE", data=nouveauPokedex)
plt.xticks(rotation= 90)
plt.xlabel('TYPE_1')
plt.ylabel('Total ')
plt.title("POKEMONS DE TYPE_1")
plt.show()

#Visualisation des Pokemons de type 2
axe_X = sns.countplot(x="TYPE_2", hue="LEGENDAIRE", data=nouveauPokedex)
plt.xticks(rotation= 90)
plt.xlabel('TYPE_2')
plt.ylabel('Total ')
plt.title("POKEMONS DE TYPE_2")
plt.show()

#Recherche de correlation
print(nouveauPokedex.groupby('TYPE_1').agg({"POURCENTAGE_DE_VICTOIRES": "mean"}).sort_values(by = "POURCENTAGE_DE_VICTOIRES"))
corr = nouveauPokedex.loc[:,['TYPE_1','POINTS_DE_VIE','NIVEAU_ATTAQUE','NIVEAU_DEFENSE','NIVEAU_ATTAQUE_SPECIALE','NIVEAU_DEFENSE_SPECIALE','VITESSE','LEGENDAIRE','POURCENTAGE_DE_VICTOIRES']].corr()
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
plt.show()

#Decoupage en jeu d'apprentissage et jeu de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Sauvegarde du Pokedex
dataset = nouveauPokedex
dataset.to_csv("datas/dataset.csv", sep='\t')
