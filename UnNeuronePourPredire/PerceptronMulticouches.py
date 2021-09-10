# Pour utiliser tensorflow 1 et les placeholder avec le 2.0
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

# Mettre en cas de RuntimeError: tf.placeholder() is not compatible with eager execution.
tf.disable_v2_behavior()

# -------------------------------------
#    DONNEES D'APPRENTISSAGE
# -------------------------------------

# On transforme les donnees en décimales

valeurs_entrees_X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
valeurs_a_predire_Y = [[0.], [1.], [1.], [0.]]

# -------------------------------------
#    PARAMETRES DU RESEAU
# -------------------------------------
# Variable TensorFLow correspondant aux valeurs neurones d'entrée
tf_neurones_entrees_X = tf.placeholder(tf.float32, [None, 2])

# Variable TensorFlow correspondant au neurone de sortie (prédiction reele)
tf_valeurs_reelles_Y = tf.placeholder(tf.float32, [None, 1])

# Nombre de neurones dans la couche cachée
nbr_neurones_couche_cachee = 2

# POIDS
# Les premiers sont au nombres de 4 : 2 entrées (X1 et X2) et 2 poids par entrées
poids = tf.Variable(tf.random_normal([2, 2]), tf.float32)

# les poids de la couche cachee sont au nombre de 2 : 2 entrée (H1 et H2) et 1 poids par entrée
poids_couche_cachee = tf.Variable(tf.random_normal([2, 1]), tf.float32)

# Premier biais comporte 2 poids
biais = tf.Variable(tf.zeros([2]))

# Le second biais comporte 1 poids
biais_couche_cachee = tf.Variable(tf.zeros([1]))

# Calcul de l'activation de la première couche
# calcul de la somme pondérée (tf.matmul) à l'aide des données X1, X2, W11,W12,W31,W41 et du bais
# puis application de la fonction sigmoide (tf.sigmoid)
activation = tf.sigmoid(tf.matmul(tf_neurones_entrees_X, poids) + biais)

# Calcul de l'activation de la couche cachée
# calcul de la somme pondérée (tf.matmul) à l'aide des données H1, H2, W12,W21 et du bais
# puis application de la fonction sigmoide (tf.sigmoid)
activation_couche_cachee = tf.sigmoid(tf.matmul(activation, poids_couche_cachee) + biais_couche_cachee)

# Fonction d'erreur de moyenne quadratique MSE
fonction_erreur = tf.reduce_sum(tf.pow(tf_valeurs_reelles_Y - activation_couche_cachee, 2))

# Descente de gradient avec un taux d'apprentissage fixé à 0.1
optimiseur = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(fonction_erreur)

# Nombre d'epochs
epochs = 100000

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

print("--- VERIFICATIONS ----")

for i in range(0, 4):
    print("Observation:" + str(valeurs_entrees_X[i]) + " - Attendu: " + str(
        valeurs_a_predire_Y[i]) + " - Prediction: " + str(
        session.run(activation_couche_cachee, feed_dict={tf_neurones_entrees_X: [valeurs_entrees_X[i]]})))

# Observation:[0.0, 0.0] - Attendu: [0.0] - Prediction: [[0.00794011]]
# Observation:[0.0, 1.0] - Attendu: [1.0] - Prediction: [[0.9925213]]
# Observation:[1.0, 0.0] - Attendu: [1.0] - Prediction: [[0.99089503]]
# Observation:[1.0, 1.0] - Attendu: [0.0] - Prediction: [[0.00711811]]

session.close()
