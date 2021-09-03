from math import exp

from numpy import array, random

# -------------------------------------
#    OBSERVATIONS ET PREDICTIONS
# -------------------------------------

observations_entrees = array([
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]
])

predictions = array([[0], [1], [0], [0]])

# --------------------------------------
#        PARAMETRAGE DU PERCEPTRON
# --------------------------------------

# Génération des poids dans l'interval [-1;1]
random.seed(1)
borneMin = -1
borneMax = 1

# Neurone 1 de la couche 1
w11 = (borneMax - borneMin) * random.random() + borneMin
# Neurone 2 de la couche 1
w21 = (borneMax - borneMin) * random.random() + borneMin
# Neurone 3 de la couche 1
w31 = (borneMax - borneMin) * random.random() + borneMin

# Le biais
biais = 1
wb = 0

# Taux d'apprentissage
txApprentissage = 0.1

# Nombres d'epoques
epochs = 300


# --------------------------------------
#       FONCTIONS UTILES
# --------------------------------------


def somme_ponderee(X1, W11, X2, W21, B, WB):
    return B * WB + (X1 * W11 + X2 * W21)


def fonction_activation_sigmoide(valeur_somme_ponderee):
    return 1 / (1 + exp(-valeur_somme_ponderee))


def fonction_activation_relu(valeur_somme_ponderee):
    return max(0, valeur_somme_ponderee)


def erreur_lineaire(valeur_attendue, valeur_predite):
    return valeur_attendue - valeur_predite


def calcul_gradient(valeur_entree, prediction, erreur):
    return -1 * erreur * prediction * (1 - prediction) * valeur_entree


def calcul_valeur_ajustement(valeur_gradient, taux_apprentissage):
    return valeur_gradient * taux_apprentissage


def calcul_nouveau_poids(valeur_poids, valeur_ajustement):
    return valeur_poids - valeur_ajustement


def calcul_mse(predictions_realisees, f_predictions_attendues):
    i = 0
    somme = 0
    for _ in f_predictions_attendues:
        difference = f_predictions_attendues[i] - predictions_realisees[i]
        carre_difference = difference * difference
        somme = somme + carre_difference
    moyenne_quadratique = 1 / (len(f_predictions_attendues)) * somme
    return moyenne_quadratique


# --------------------------------------
#    APPRENTISSAGE
# --------------------------------------

for epoch in range(0, epochs):
    print("EPOCH (" + str(epoch) + "/" + str(epochs) + ")")
    predictions_realisees_durant_epoch = []
    predictions_attendues = []
    numObservation = 0
    for observation in observations_entrees:
        # Chargement de la couche d'entrée
        x1 = observation[0]
        x2 = observation[1]

        # Valeur de prédiction attendue
        valeur_attendue = predictions[numObservation][0]

        # Etape 1 : Calcul de la somme ponderee
        valeur_somme_ponderee = somme_ponderee(x1, w11, x2, w21, biais, wb)

        # Etape 2 : Application de la fonction d'activation
        valeur_predite = fonction_activation_sigmoide(valeur_somme_ponderee)

        # Etape 3 : Calcul de l'erreur
        valeur_erreur = erreur_lineaire(valeur_attendue, valeur_predite)

        # Mise à jour du poids 1
        # Calcul du gradient de la valeur d'ajustement et du nouveau poids
        gradient_W11 = calcul_gradient(x1, valeur_predite, valeur_erreur)
        valeur_ajustement_W11 = calcul_valeur_ajustement(gradient_W11, txApprentissage)
        w11 = calcul_nouveau_poids(w11, valeur_ajustement_W11)

        # Mise à jour du poids 2
        gradient_W21 = calcul_gradient(x2, valeur_predite, valeur_erreur)
        valeur_ajustement_W21 = calcul_valeur_ajustement(gradient_W21, txApprentissage)
        w21 = calcul_nouveau_poids(w21, valeur_ajustement_W21)

        # Mise à jour du poids du biais
        gradient_Wb = calcul_gradient(biais, valeur_predite, valeur_erreur)
        valeur_ajustement_Wb = calcul_valeur_ajustement(gradient_Wb, txApprentissage)
        wb = calcul_nouveau_poids(wb, valeur_ajustement_Wb)

        print("     EPOCH (" + str(epoch) + "/" + str(epochs) + ") -  Observation: " + str(
            numObservation + 1) + "/" + str(len(observations_entrees)))

        # Stockage de la prediction realisee:
        predictions_realisees_durant_epoch.append(valeur_predite)
        predictions_attendues.append(predictions[numObservation][0])

        # Passage à l'observation suivante
        numObservation = numObservation + 1

    MSE = calcul_mse(predictions_realisees_durant_epoch, predictions)
    print("MSE : " + str(MSE))
