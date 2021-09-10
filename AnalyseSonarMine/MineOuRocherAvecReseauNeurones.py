import numpy as np
import pandas as pnd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------
# CHARGEMENT DES OBSERVATIONS
# ---------------------------------------------

observations = pnd.read_csv("datas/sonar.all-data.csv")

# ---------------------------------------------
# PREPARATION DES DONNEES
# ---------------------------------------------

print("Nbr colonnes: ", len(observations.columns))
# On ne prend que les données issues du sonar pour l'apprentissage
X = observations[observations.columns[0:60]].values

# On ne prend que les libellé
y = observations[observations.columns[60]]

# On encode : Les mines sont égales à 0 et les rochers égaux à 1
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# On ajoute un encodage pour créer des classes :
# Si c'est une mine [1,0]
# Si c'est un rocher [0,1]
n_labels = len(y)
n_unique_labels = len(np.unique(y))
one_hot_encode = np.zeros((n_labels, n_unique_labels))
one_hot_encode[np.arange(n_labels), y] = 1
Y = one_hot_encode

# Verification en prenant les enregistrement 0 et 97
print("Classe Rocher:", int(Y[0][1]))
print("Classe Mine :", int(Y[97][1]))
