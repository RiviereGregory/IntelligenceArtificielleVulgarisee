import numpy as np
import pandas as pnd
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Mettre en cas de RuntimeError: tf.placeholder() is not compatible with eager execution.
tf.disable_v2_behavior()

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

# ---------------------------------------------
# CREATION DES JEUX D'APPRENTISSAGE ET DE TEST
# ---------------------------------------------

# On mélange les observations
X, Y = shuffle(X, Y, random_state=1)

# Creation des jeux d'apprentissage et de tests
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.07, random_state=42)

# ---------------------------------------------
# PARAMETRAGE DU RESEAU DE  NEURONES
# ---------------------------------------------

epochs = 300
nombre_neurones_entree = 60
nombre_neurones_sortie = 2
taux_apprentissage = 0.01

# Variable TensorFLow correspondant aux 60 valeurs des neurones d'entrée
tf_neurones_entrees_X = tf.placeholder(tf.float32, [None, 60])

# Variable TensorFlow correspondant au 2 neurones de sortie
tf_valeurs_reelles_Y = tf.placeholder(tf.float32, [None, 2])

poids = {
    # 60 neurones d'entrées vers 24 Neurones de la couche cachée
    'couche_entree_vers_cachee': tf.Variable(tf.random_uniform([60, 24], minval=-0.3, maxval=0.3), tf.float32),

    # 24 neurones de la couche cachée vers 2 de la couche de sortie
    'couche_cachee_vers_sortie': tf.Variable(tf.random_uniform([24, 2], minval=-0.3, maxval=0.3), tf.float32),
}

poids_biais = {
    # 1 biais de la couche d'entrée vers les 24 neurones de la couche cachée
    'poids_biais_couche_entree_vers_cachee': tf.Variable(tf.zeros([24]), tf.float32),

    # 1 biais de la couche cachée vers les 2 neurones de la couche de sortie
    'poids_biais_couche_cachee_vers_sortie': tf.Variable(tf.zeros([2]), tf.float32),
}


# ---------------------------------------------
# FONCTION DE  CREATION DU RESEAU DE NEURONES
# ---------------------------------------------

def reseau_neurones_multicouches(observations_en_entrees, poids_f, poids_biais_f):
    # Calcul de l'activation de la première couche
    premiere_activation = tf.sigmoid(
        tf.matmul(tf_neurones_entrees_X, poids_f['couche_entree_vers_cachee']) + poids_biais_f[
            'poids_biais_couche_entree_vers_cachee'])

    # Calcul de l'activation de la seconde couche
    activation_couche_cachee = tf.sigmoid(
        tf.matmul(premiere_activation, poids_f['couche_cachee_vers_sortie']) + poids_biais_f[
            'poids_biais_couche_cachee_vers_sortie'])

    return activation_couche_cachee


# ---------------------------------------------
# CREATION DU RESEAU DE NEURONES
# ---------------------------------------------
reseau = reseau_neurones_multicouches(tf_neurones_entrees_X, poids, poids_biais)

# ---------------------------------------------
# ERREUR ET OPTIMISATION
# ---------------------------------------------

# Fonction d'erreur de moyenne quadratique MSE
fonction_erreur = tf.reduce_sum(tf.pow(tf_valeurs_reelles_Y - reseau, 2))

# Descente de gradient avec un taux d'apprentissage fixé à 0.1
optimiseur = tf.train.GradientDescentOptimizer(learning_rate=taux_apprentissage).minimize(fonction_erreur)
