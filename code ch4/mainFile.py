#-----------------------------------------------------------------------------------------
# @Auteur : Aurélien Vannieuwenhuyze
# @Entreprise : Junior Makers Place
# @Livre
# @Chapitre : 04 - Un peu de statistiques descriptives pour comprendre les données
#
# Modules necessaires : 
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#   JMPStatistiques (copier le fichier dans votre projet au même niveau que ce fichier)
#
# Pour installer un module : 
#   Cliquer sur le menu File > Settings > Project:nom_du_projet > Project interpreter > bouton +
#   Dans la zone de recherche en haut à gauche saisir le nom du module
#   Choisir la version en bas à droite
#   Cliquer sur le bouton install situé en bas à gauche
#-----------------------------------------------------------------------------------------


import pandas as pnd
import JMPStatistiques as jmp
import numpy as np

#--- CREATION D'UN DATAFRAME ----
observations = pnd.DataFrame({'NOTES':np.array([3,19,10,15,14,12,9,8,11,12,11,12,13,11,14,16])})

#--- ANALYSE D'UNE FEATURE ---
stats = jmp.JMPStatistiques(observations['NOTES'])
stats.analyseFeature()