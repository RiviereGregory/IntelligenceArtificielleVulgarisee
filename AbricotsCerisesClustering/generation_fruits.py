# Pour avoir les calibres des fruits :
# http://www.crenoexpert.fr/flipbooks/expproduit/TABLEAUX-CALIBRES-FRUITS-2.pdf

####################################
# ------ IMPORT DES MODULES ------ #
####################################
import random

import pandas as pnd

####################################
# ------- CARACTERISTIQUES ------- #
####################################

# CERISES
# [diam min, diam max, poids min, poids max]
caracteristiquesCerises = [[17, 19, 1, 5], [20, 21, 5, 6], [22, 23, 6, 7], [24, 25, 7, 8.5], [26, 27, 8.5, 10],
                           [28, 29, 10, 11.5]]

# ABRICOTS
# [diam min, diam max, poids moy]
# Cas 1 :
# caracteristiquesAbricots = [[40, 44, 41], [45, 49, 54], [50, 54, 74], [55, 59, 100]]

# Cas 2 :
caracteristiquesAbricots = [[35,39,27],[40,44,41],[45,49,54],[50,54,74],[55,59,100]]

####################################
# ---- GENERATION DES DONNEES ---- #
####################################
# [DIAMETRE, POIDS]
nombreObservations = 2000

# 1 Generation des cerises
cerises = []
random.seed()
for iteration in range(nombreObservations):
    # choix au hazard d'une caracteristique
    cerise = random.choice(caracteristiquesCerises)
    # Generation d'un diametre
    diametre = round(random.uniform(cerise[0], cerise[1]), 2)
    # Generation d'un poids
    poids = round(random.uniform(cerise[2], cerise[3]), 2)
    print("Cerise " + str(iteration) + " " + str(cerise) + " : " + str(diametre) + " - " + str(poids))
    cerises.append([diametre, poids])

# 2 Generation des abricots
abricots = []
random.seed()
for iteration in range(nombreObservations):
    # choix au hazard d'une caracteristique
    abricot = random.choice(caracteristiquesAbricots)
    # Generation d'un diametre
    diametre = round(random.uniform(abricot[0], abricot[1]), 2)
    # Generation d'un poids
    borneMinPoids = abricot[2] / 1.10
    borneMaxPoids = abricot[2] * 1.10
    poids = round(random.uniform(borneMinPoids, borneMaxPoids), 2)
    print("Abricot " + str(iteration) + " " + str(abricot) + " : " + str(diametre) + " - " + str(poids))
    abricots.append([diametre, poids])

# 3 Constitution des observations
fruits = cerises + abricots
print(fruits)

# 4 Mélange des observations
random.shuffle(fruits)

# 5 Sauvegarde des observations dans un fichier
dataFrame = pnd.DataFrame(fruits)
dataFrame.to_csv("datas/fruits.csv", index=False, header=False)
