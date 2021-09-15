import numpy as np
import pandas as pnd
from matplotlib import pyplot as plt

# Définition de la longueur et de la largeur de l'image
LONGUEUR_IMAGE = 28
LARGEUR_IMAGE = 28

# Chargement des données d'entrainement
observations_entrainement = pnd.read_csv('datas/fashion-mnist_train.csv')

# On exclut la première colonne (les labels) pour constituer un tableau de pixels
X = np.array(observations_entrainement.iloc[:, 1:])

premiere_image = X[0]
premiere_image = premiere_image.reshape([LONGUEUR_IMAGE, LARGEUR_IMAGE])
plt.imshow(premiere_image)
plt.show()
