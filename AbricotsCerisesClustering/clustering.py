####################################
# ------ IMPORT DES MODULES ------ #
####################################
import matplotlib.pyplot as plt
import pandas as pnd
from joblib import load
from sklearn.cluster import KMeans

################################
# ------- VISUALISTION ------- #
################################

# 1 Chargement des données
fruits = pnd.read_csv("datas/fruits.csv", names=['DIAMETRE', 'POIDS'], header=None)

# 2 Visualisation graphique des données
fruits.plot.scatter(x="DIAMETRE", y="POIDS")
plt.show()

# Determination manuelle des 2 clusters Cerises (en bas à gauche) et Abricots (en haut à droite)

##############################
# ------- CLUSTERING ------- #
##############################
# 1 Apprentissage avec l'algorithme K-Mean
modele = KMeans(n_clusters=2)
modele.fit(fruits)

# 2 Predictions
predictions_kmeans = modele.predict(fruits)

# 3 Affichage de la clusterisation
plt.scatter(fruits.DIAMETRE, fruits.POIDS, c=predictions_kmeans, s=50, cmap='viridis')
plt.xlabel("DIAMETRE")
plt.ylabel("POIDS")

# 4 Affichage des centroïdes
centers = modele.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()


# Sauvegarde du modèle ( A décommenter si besoin)
# from joblib import dump
#
# dump(modele, 'modeles/kmean.joblib')

########################################
# -- Realisation de classifications -- #
########################################
# fonction
def prediction(num_cluster):
    if int(num_cluster) == 0:
        print("C'est un abricot !")
    else:
        print("C'est une cerise ! ")


# chargement du modèle
modele_load = load('modeles/kmean.joblib')

# données test
# CERISE: 26.98 mm de diametre ,8.75 grammes
# ABRICOT: 55.7  mm de diametre , 102.16 grammes

cerise = [[26.98, 8.75]]
numCluster = modele_load.predict(cerise)
print("Numero de cluster des cerises: " + str(numCluster))
prediction(numCluster)

abricot = [[55.7, 102.16]]
numCluster = modele_load.predict(abricot)
print("Numero de cluster des abricots: " + str(numCluster))
prediction(numCluster)
