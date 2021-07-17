import pandas as pnd
from matplotlib import pyplot as plt
# Algorithme de "boosting"
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# fonction de calcul du précision accuracy_score pour la classification
from sklearn.metrics import accuracy_score
# Pour tester tous les hyperparamtres du SVM
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import JMPStatistiques as jmp

# ------- Préparation des données -------- #
observations = pnd.read_csv("datas/sonar.all-data.csv")
print(observations.columns.values)
# ['0.0200' '0.0371' '0.0428' '0.0207' '0.0954' '0.0986' '0.1539' '0.1601'
#  '0.3109' '0.2111' '0.1609' '0.1582' '0.2238' '0.0645' '0.0660' '0.2273'
#  '0.3100' '0.2999' '0.5078' '0.4797' '0.5783' '0.5071' '0.4328' '0.5550'
#  '0.6711' '0.6415' '0.7104' '0.8080' '0.6791' '0.3857' '0.1307' '0.2604'
#  '0.5121' '0.7547' '0.8537' '0.8507' '0.6692' '0.6097' '0.4943' '0.2744'
#  '0.0510' '0.2834' '0.2825' '0.4256' '0.2641' '0.1386' '0.1051' '0.1343'
#  '0.0383' '0.0324' '0.0232' '0.0027' '0.0065' '0.0159' '0.0072' '0.0167'
#  '0.0180' '0.0084' '0.0090' '0.0032' 'R']

print(observations.shape)
# (207, 61) --> 208 observations et 62 features

# Ajout d'un nom pour chaque feature
observations = pnd.read_csv("datas/sonar.all-data.csv", names=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
                                                               "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17",
                                                               "F18", "F19",
                                                               "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27",
                                                               "F28", "F29",
                                                               "F30", "F31", "F32", "F33", "F34", "F35", "F36", "F37",
                                                               "F38", "F39",
                                                               "F40", "F41", "F42", "F43", "F44", "F45", "F46", "F47",
                                                               "F48", "F49",
                                                               "F50", "F51", "F52", "F53", "F54", "F55", "F56", "F57",
                                                               "F58", "F59",
                                                               "F60", "OBJET"])

# Desactivation du nombre maximum de colonnes du DataFrame à afficher
pnd.set_option('display.max_columns', None)

# Affichage des 10 premieres observations
print(observations.head(10))

# Transformation de la caractéristique Objet
# 1 pour type mine
# 0 pour type rocher
observations['OBJET'] = (observations['OBJET'] == 'M').astype(int)

# Vérification qu'il ne manque pas d'information
print(observations.info())
# --> manque aucune valeurs


# ------- Analyse données -------- #

# Vérification du nombre de mines et de rochers
print(observations.groupby("OBJET").size())
# 111 Mines et 97 rochers

# Calcul moyenne,ecart type, min , max et quartiles
print(observations.describe())

# Recherche des valeurs extrêmes à l'aide de la librairie JMPStatistiques
stats = jmp.JMPStatistiques(observations['F1'])
stats.analyseFeature()

# Création graph boite moustache
observations.plot.box(figsize=(20, 10), xticks=[])
# Info graph
plt.title('Détection des valeurs extrêmes')
plt.xlabel('Les 60 frèquences')
plt.ylabel('Puissancs du signal')
plt.show()
# --> les valeurs extrêmes sont les rond noir en dehors des moustaches

# ------- Choix d'un modèle de prédiction -------- #
array = observations.values
# convertion en type décimal
X = array[:, 0:-1].astype(float)

# Choix de la dernière colonne commme feature de prédiction
Y = array[:, -1]

# Création jeux d'apprentissage et de test
percentage_donnees_test = 0.2
X_APPRENTISSAGE, X_VALIDATION, Y_APPRENTISSAGE, Y_VALIDATION = train_test_split(X, Y, test_size=percentage_donnees_test,
                                                                                random_state=42)
# REGRESSION LOGISTIQUE
regression_logistique = LogisticRegression()
regression_logistique.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = regression_logistique.predict(X_VALIDATION)
print("Regression logistique: " + str(accuracy_score(predictions, Y_VALIDATION)))

# ARBRE DE DECISION
arbre_decision = DecisionTreeClassifier()
arbre_decision.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = arbre_decision.predict(X_VALIDATION)
print("Arbre de décision:  " + str(accuracy_score(predictions, Y_VALIDATION)))

# FORET ALEATOIRES
foret_aleatoire = RandomForestClassifier()
foret_aleatoire.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = foret_aleatoire.predict(X_VALIDATION)
print("Foret aléatoire: " + str(accuracy_score(predictions, Y_VALIDATION)))

# K PLUS PROCHES VOISINS
knn = KNeighborsClassifier()
knn.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = knn.predict(X_VALIDATION)
print("K plus proche voisins: " + str(accuracy_score(predictions, Y_VALIDATION)))

# MACHINE VECTEURS DE SUPPORT
SVM = SVC(gamma='auto')
SVM.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = SVM.predict(X_VALIDATION)
print("Machine vecteurs de support: " + str(accuracy_score(predictions, Y_VALIDATION)))

# Regression logistique: 0.7857142857142857
# Arbre de décision:  0.5952380952380952
# Foret aléatoire: 0.8333333333333334
# K plus proche voisins: 0.8571428571428571
# Machine vecteurs de support: 0.8333333333333334

# Optimisation du SVM (MACHINE VECTEURS DE SUPPORT)
# Définition d'une plage de valeurs à tester
penalite = [{'C': range(1, 100)}]

# Tests avec 5 échantillon de Validation Croisée
recherche_optimisations = GridSearchCV(SVC(), penalite, cv=5)
recherche_optimisations.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

print("Le meilleur paramètre est :")
print()
print(recherche_optimisations.best_params_)
print()
# Le meilleur paramètre est : {'C': 35}

# MACHINE A VECTEURS DE SUPPORT OPTIMISE
SVM = SVC(C=35, gamma='auto')
SVM.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = SVM.predict(X_VALIDATION)
print("Machine à vecteurs de support optimisé: " + str(accuracy_score(predictions, Y_VALIDATION)))
# Machine à vecteurs de support optimisé: 0.9047619047619048

# --> Le meilleur algo dans ce cas est le SVM (MACHINE VECTEURS DE SUPPORT)

# Test du gradient boosting pour voir s'il est le meilleur
gradientBoosting = GradientBoostingClassifier()
gradientBoosting.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = SVM.predict(X_VALIDATION)
print("GRADIENT BOOSTING: " + str(accuracy_score(predictions, Y_VALIDATION)))
# --> GRADIENT BOOSTING: 0.9047619047619048
# Sans optimistaion le RADIENT BOOSTING est le plus efficace
