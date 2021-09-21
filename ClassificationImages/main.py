import numpy as np
import pandas as pnd
from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical

# Définition de la longueur et de la largeur de l'image
LONGUEUR_IMAGE = 28
LARGEUR_IMAGE = 28

# Chargement des données d'entrainement
observations_entrainement = pnd.read_csv('datas/fashion-mnist_train.csv')

# On exclut la première colonne (les labels) pour constituer un tableau de pixels
X = np.array(observations_entrainement.iloc[:, 1:])

# premiere_image = X[0]
# premiere_image = premiere_image.reshape([LONGUEUR_IMAGE, LARGEUR_IMAGE])
# plt.imshow(premiere_image)
# plt.show()

# On crée un tableau de catégories à l'aide du module Keras
y = to_categorical(np.array(observations_entrainement.iloc[:, 0]))

# Répartition des données d'entrainement en données d'apprentissage et donnée de validation
# 80% de donnée d'apprentissage et 20% de donnée de validation
X_apprentissage, X_validation, y_apprentissage, y_validation = train_test_split(X, y, test_size=0.2, random_state=13)

# On redimensionne les images au format 28*28 et on réalise un scaling sur les données des pixels
X_apprentissage = X_apprentissage.reshape(X_apprentissage.shape[0], LARGEUR_IMAGE, LONGUEUR_IMAGE, 1)
X_apprentissage = X_apprentissage.astype('float32')
X_apprentissage /= 255

# On fait la même chose avec les données de validation
X_validation = X_validation.reshape(X_validation.shape[0], LARGEUR_IMAGE, LONGUEUR_IMAGE, 1)
X_validation = X_validation.astype('float32')
X_validation /= 255
