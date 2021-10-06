import numpy as np
from matplotlib import pyplot as plt
from mnist import MNIST

# Chargement des images
emnist_data = MNIST(path='datas\\', return_type='numpy')
emnist_data.select_emnist('letters')
Images, Libelles = emnist_data.load_training()

# Vérification des données
print("Nombre d'images = " + str(len(Images)))
print("Nombre de libellés = " + str(len(Libelles)))
# Nombre d'images = 124800
# Nombre de libellés = 124800

# Conversion des images et libellés en tableau numpy
Images = np.asarray(Images)
Libelles = np.asarray(Libelles)

# Dimension des images de travail et d'apprentissage
longueurImage = 28
largeurImage = 28

# Les images sont sous forme d'un tableau de 124800 lignes et 784 colonnes
# On les transforme en un tableau comportant 124800 lignes contenant un tableau de 28*28 colonnes
print("Transformation des tableaux d'images...")
Images = Images.reshape(124800, largeurImage, longueurImage)
Libelles = Libelles.reshape(124800, 1)

print("Affichage de l'image N°70000...")
plt.imshow(Images[70000])
plt.show()

print(Libelles[70000])

# En informatique, les index des listes doivent commencer à zéro...")
Libelles = Libelles - 1

print("Libellé de l'image N°70000...")
print(Libelles[70000])
