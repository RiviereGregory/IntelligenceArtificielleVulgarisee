# Pour utiliser tensorflow 1 et les placeholder avec le 2.0
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

# -------------------------------------
#    DONNEES D'APPRENTISSAGE
# -------------------------------------

valeurs_entrees_X = [[1., 0.], [1., 1.], [0., 1.], [0., 0.]]
valeurs_a_predire_Y = [[0.], [1.], [0.], [0.]]

# -------------------------------------
#    PARAMETRES DU RESEAU
# -------------------------------------

# Variable TensorFLow correspondant aux valeurs neurones d'entrée
tf_neurones_entrees_X = tf.placeholder(tf.float32, [None, 2])

# Variable TensorFlow correspondant au neurone de sortie (prédiction reele)
tf_valeurs_reelles_Y = tf.placeholder(tf.float32, [None, 1])

# -- Poids --
# Création d'une variable TensorFlow de type tableau
# contenant 2 entrées ayant chacune un 1 poids [2,1]
# Ces valeurs sont initialisées aux hasard
poids = tf.Variable(tf.random_normal([2, 1]), tf.float32)

# -- Biais initialisée à 0 --
biais = tf.Variable(tf.zeros([1, 1]), tf.float32)

# La somme pondérée est en fait une multiplication de matrice
# entre les valeur en entrées X et les différents poids
# la fonction matmul se charge de faire cette multiplication
sommeponderee = tf.matmul(tf_neurones_entrees_X, poids)

# Ajout du biais à la somme ponderee
sommeponderee = tf.add(sommeponderee, biais)

# Fonction d'activation de type sigmoide permettant de calculer la prédiction
prediction = tf.sigmoid(sommeponderee)

# Fonction d'erreur de moyenne quadratique MSE
fonction_erreur = tf.reduce_sum(tf.pow(tf_valeurs_reelles_Y - prediction, 2))

# Descente de gradient avec un taux d'apprentissage fixé à 0.1
optimiseur = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(fonction_erreur)

# -------------------------------------
#    APPRENTISSAGE
# -------------------------------------

# Nombre d'epochs
epochs = 10000

# Initialisation des variable
init = tf.global_variables_initializer()

# Demarrage d'une session d'apprentissage
session = tf.Session()
session.run(init)

# Pour la réalisation du graphique pour la MSE
Graphique_MSE = []

# Pour chaque epoch
for i in range(epochs):
    # Realisation de l'apprentissage avec mise à jour des poids
    session.run(optimiseur,
                feed_dict={tf_neurones_entrees_X: valeurs_entrees_X, tf_valeurs_reelles_Y: valeurs_a_predire_Y})

    # Calculer l'erreur
    MSE = session.run(fonction_erreur,
                      feed_dict={tf_neurones_entrees_X: valeurs_entrees_X, tf_valeurs_reelles_Y: valeurs_a_predire_Y})

    # Affichage des informations
    Graphique_MSE.append(MSE)
    print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: " + str(MSE))

# Affichage graphique

plt.plot(Graphique_MSE)
plt.ylabel('MSE')
plt.show()
