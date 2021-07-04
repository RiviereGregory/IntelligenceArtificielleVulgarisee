# ***********************
# IMPORT DES MODULES
# ***********************
import os

# ***********************
# ANALYSE DES DONNEES
# ***********************

# Récupération des fichiers contenus dans datas
listeDeFichiers = os.listdir("datas")

# Nom des fichiers
for fichier in listeDeFichiers:
    print(fichier)
