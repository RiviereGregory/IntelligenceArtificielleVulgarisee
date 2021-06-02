#-----------------------------------------------------------------------------------------
# @Auteur : Aurélien Vannieuwenhuyze
# @Entreprise : Junior Makers Place
# @Livre
# @Chapitre : 08 - Réaliser une bonne classification n'est pas une option
#
# Modules necessaires :
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#   MATPLOTLIB 3.0.3
#   SCIKIT-LEARN : 0.21.0
#   JMPStatistiques (copier le fichier dans votre projet au même niveau que ce fichier)
#
# Pour installer un module :
#   Cliquer sur le menu File > Settings > Project:nom_du_projet > Project interpreter > bouton +
#   Dans la zone de recherche en haut à gauche saisir le nom du module
#   Choisir la version en bas à droite
#   Cliquer sur le bouton install situé en bas à gauche
#-----------------------------------------------------------------------------------------

#Aquisition des données
import pandas as pnd
observations = pnd.read_csv("datas/sonar.all-data.csv")


import pandas as pnd
observations = pnd.read_csv("datas/sonar.all-data.csv", names=["F1","F2","F3","F4","F5","F6","F7","F8","F9",
                                                      "F10","F11","F12","F13","F14","F15","F16","F17","F18","F19",
                                                      "F20","F21","F22","F23","F24","F25","F26","F27","F28","F29",
                                                      "F30","F31","F32","F33","F34","F35","F36","F37","F38","F39",
                                                      "F40","F41","F42","F43","F44","F45","F46","F47","F48","F49",
                                                      "F50","F51","F52","F53","F54","F55","F56","F57","F58","F59",
                                                      "F60","OBJET"])

observations['OBJET'] = (observations['OBJET']=='M').astype(int)

#Pour chaque caractéristique on cherche les numéros de ligne correspondant
#a une donnée extreme
import numpy as np

#On crée une liste chargée de contenir les numéros de lignes correspondants
#a une valeur extreme
num_lignes = []

#On parcours toutes les 60 caracteristiques
for caracteristique in observations.columns.tolist():
    #Pour une caracteristique : calcul des percentile
    Q1 = np.percentile(observations[caracteristique],25)
    Q3 = np.percentile(observations[caracteristique],75)
    #Calcul de la borne
    donnee_extreme = 1.5*(Q3-Q1)
    #Si la donnée est inférieure ou supérieure à la borne on récupere sont numéro de ligne et on l'ajoute à la liste
    liste_donnees_extremes = observations[(observations[caracteristique]<Q1-donnee_extreme) | (observations[caracteristique]>Q3+donnee_extreme)].index
    num_lignes.extend(liste_donnees_extremes)



#On ordonne la liste par ordre croissant
num_lignes.sort()


#On crée une liste contenant les numéros de lignes à supprimer
num_lignes_a_supprimer=[]


#On parcours l'ensemble des numéros de lignes
for ligne in num_lignes :
    #Pour une ligne, on récupere son numéro
    num_ligne = ligne
    #On calcul le nombre de fois où apparait ce numéro de ligne
    #dans l'ensemble des numeros de lignes
    nbr_valeurs_extremes = num_lignes.count(num_ligne)

    #Si le nombre d'erreur est supérieur 7 alors on ajoute le numéro de la
    #ligne à la liste des lignes à supprimer
    if (nbr_valeurs_extremes>7):
        num_lignes_a_supprimer.append(num_ligne)



#On supprime les doublons
num_lignes_a_supprimer = list(set(num_lignes_a_supprimer))


#On supprime ensuite les lignes dans le dataframe
print(num_lignes_a_supprimer)
print("Nombre de lignes à supprimer = "+str(len(num_lignes_a_supprimer)))
observations = observations.drop(num_lignes_a_supprimer,axis=0)
print()
print()



#Utilisation du module Scikit-Learn
from sklearn.model_selection import train_test_split
array = observations.values

#Convertion des données en type decimal
X = array[:,0:-1].astype(float)

#On choisi la dernière colonne comme feature de prédiction
Y = array[:,-1]

#Création des jeux d'apprentissage et de tests
percentage_donnees_test = 0.2
X_APPRENTISSAGE, X_VALIDATION, Y_APPRENTISSAGE, Y_VALIDATION = train_test_split(X, Y, test_size=percentage_donnees_test, random_state=42)

#Import des algorithmes et de la fonction de calcul du précision
#accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Suppression des erreurs de type warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#REGRESSION LOGISTIQUE
regression_logistique = LogisticRegression()
regression_logistique.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = regression_logistique.predict(X_VALIDATION)
print("Regression logistique: "+str(accuracy_score(predictions, Y_VALIDATION)))

#ARBRE DE DECISION
arbre_decision = DecisionTreeClassifier()
arbre_decision.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = arbre_decision.predict(X_VALIDATION)
print("Arbre de décision:  "+str(accuracy_score(predictions, Y_VALIDATION)))


#FORET ALEATOIRES
foret_aleatoire= RandomForestClassifier()
foret_aleatoire.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = foret_aleatoire.predict(X_VALIDATION)
print("Foret aléatoire: "+str(accuracy_score(predictions, Y_VALIDATION)))


#K PLUS PROCHES VOISINS
knn = KNeighborsClassifier()
knn.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = knn.predict(X_VALIDATION)
print("K plus proche voisins: "+str(accuracy_score(predictions, Y_VALIDATION)))


#MACHINE VECTEURS DE SUPPORT
SVM = SVC(gamma='auto')
SVM.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = SVM.predict(X_VALIDATION)
print("Machine vecteurs de support: "+str(accuracy_score(predictions, Y_VALIDATION)))


from sklearn.model_selection import GridSearchCV

#Définition d'une plage de valeurs à tester
penalite = [{'C': range(1,100)}]


#Tests avec 5 échantillon de Validation Croisée
recherche_optimisations = GridSearchCV(SVC(), penalite, cv=5)
recherche_optimisations.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

print("Le meilleur paramètre est :")
print()
print(recherche_optimisations.best_params_)
print()


#MACHINE A VECTEURS DE SUPPORT OPTIMISE
SVM = SVC(C=98, gamma='auto')
SVM.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = SVM.predict(X_VALIDATION)
print("Machine à vecteurs de support optimisé: "+str(accuracy_score(predictions, Y_VALIDATION)))