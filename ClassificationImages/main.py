import numpy as np
import pandas as pnd
import tensorflow.keras as keras
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
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

# Preparation des données de test
observations_test = pnd.read_csv('datas/fashion-mnist_test.csv')

X_test = np.array(observations_test.iloc[:, 1:])
y_test = to_categorical(np.array(observations_test.iloc[:, 0]))

X_test = X_test.reshape(X_test.shape[0], LARGEUR_IMAGE, LONGUEUR_IMAGE, 1)
X_test = X_test.astype('float32')
X_test /= 255

# ----------------------- CNN 1 ------------------------

# On spécifie les dimensions de l'image d'entree
dimentionImage = (LARGEUR_IMAGE, LONGUEUR_IMAGE, 1)

# On crée le réseau de neurones couche par couche
reseauNeurone1Convolution = Sequential()

# 1- Ajout de la couche de convolution comportant
#  Couche cachée de 32 neurones
#  Un filtre de 3x3 (Kernel) parourant l'image
#  Une fonction d'activation de type ReLU (Rectified Linear Activation)
#  Une image d'entrée de 28px * 28 px
reseauNeurone1Convolution.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=dimentionImage))

# 2- Définition de la fonction de pooling avec un filtre de 2px sur 2 px
reseauNeurone1Convolution.add(MaxPooling2D(pool_size=(2, 2)))

# 3- Ajout d'une fonction d'ignorance (ignorer certains neurones pour éviter le surapprentissage)
reseauNeurone1Convolution.add(Dropout(0.2))

# 4 - On transforme en une seule ligne
reseauNeurone1Convolution.add(Flatten())

# 5 - Ajout d'un reseau de neuronne composé de 128 neurones avec une fonction d'activation de type Relu
reseauNeurone1Convolution.add(Dense(128, activation='relu'))

# 6 - Ajout d'un reseau de neuronne composé de 10 neurones avec une fonction d'activation de type softmax
reseauNeurone1Convolution.add(Dense(10, activation='softmax'))

# 7 - Compilation du modèle
reseauNeurone1Convolution.compile(loss=keras.losses.categorical_crossentropy,
                                  optimizer=keras.optimizers.Adam(),
                                  metrics=['accuracy'])

# 8 - Apprentissage
historique_apprentissage = reseauNeurone1Convolution.fit(X_apprentissage, y_apprentissage,
                                                         batch_size=256,
                                                         epochs=10,
                                                         verbose=1,
                                                         validation_data=(X_validation, y_validation))

# 9 - Evaluation du modèle
evaluation = reseauNeurone1Convolution.evaluate(X_test, y_test, verbose=0)
print('Erreur :', evaluation[0])
print('Précision:', evaluation[1])

# Erreur : 0.23621676862239838
# Précision: 0.9157999753952026

# 8a - Augmentation du nombre d'images
generateur_images = ImageDataGenerator(rotation_range=8,
                                       width_shift_range=0.08,
                                       shear_range=0.3,
                                       height_shift_range=0.08,
                                       zoom_range=0.08)

nouvelles_images_apprentissage = generateur_images.flow(X_apprentissage, y_apprentissage, batch_size=256)
nouvelles_images_validation = generateur_images.flow(X_validation, y_validation, batch_size=256)

# 8b - Apprentissage
historique_apprentissage = reseauNeurone1Convolution.fit_generator(nouvelles_images_apprentissage,
                                                                   steps_per_epoch=48000 // 256,
                                                                   epochs=50,
                                                                   validation_data=nouvelles_images_validation,
                                                                   validation_steps=12000 // 256,
                                                                   use_multiprocessing=False,
                                                                   verbose=1)

# 9b - Evaluation du modèle
evaluation = reseauNeurone1Convolution.evaluate(X_test, y_test, verbose=0)
print('Erreur augmente:', evaluation[0])
print('Précision augmente:', evaluation[1])

# Erreur : 0.2179647535085678
# Précision: 0.923799991607666

# 10 - Visualisation de la phase d'apprentissage

# Données de précision (accurary)
plt.plot(historique_apprentissage.history['accuracy'])
plt.plot(historique_apprentissage.history['val_accuracy'])
plt.title('Précision du modèle')
plt.ylabel('Précision')
plt.xlabel('Epoch')
plt.legend(['Apprentissage', 'Test'], loc='upper left')
plt.show()

# Données de validation et erreur
plt.plot(historique_apprentissage.history['loss'])
plt.plot(historique_apprentissage.history['val_loss'])
plt.title('Erreur')
plt.ylabel('Erreur')
plt.xlabel('Epoch')
plt.legend(['Apprentissage', 'Test'], loc='upper left')
plt.show()

# Sauvegarde du modèle
# serialize model to JSON
model_json = reseauNeurone1Convolution.to_json()
with open("modele/modele.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
reseauNeurone1Convolution.save_weights("modele/modele.h5")
print("Modèle sauvegardé !")
