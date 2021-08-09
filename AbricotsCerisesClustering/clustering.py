####################################
# ------ IMPORT DES MODULES ------ #
####################################
import matplotlib.pyplot as plt
import pandas as pnd
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
