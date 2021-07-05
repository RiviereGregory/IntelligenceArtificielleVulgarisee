# ***********************
# IMPORT DES MODULES
# ***********************
import csv

import joblib


def recherche_informations_pokemon(num_pokemon, pokedex):
    infos_pokemon = []
    for pokemon in pokedex:
        if int(pokemon[0]) == num_pokemon:
            infos_pokemon = [pokemon[0], pokemon[1], pokemon[4], pokemon[5], pokemon[6], pokemon[7], pokemon[8],
                             pokemon[9], pokemon[10]]
            break
    return infos_pokemon


def prediction(num_pokemon1, num_pokemon2, pokedex):
    pokemon1 = recherche_informations_pokemon(num_pokemon1, pokedex)
    pokemon2 = recherche_informations_pokemon(num_pokemon2, pokedex)
    modele_prediction = joblib.load('modele/modele_pokemon.mod')
    prediction_pokemon_1 = modele_prediction.predict(
        [[pokemon1[2], pokemon1[3], pokemon1[4], pokemon1[5], pokemon1[6], pokemon1[7], pokemon1[8]]])
    prediction_pokemon_2 = modele_prediction.predict(
        [[pokemon2[2], pokemon2[3], pokemon2[4], pokemon2[5], pokemon2[6], pokemon2[7], pokemon2[8]]])
    print("COMBAT OPPOSANT : (" + str(num_pokemon1) + ") " + pokemon1[1] + " à (" + str(num_pokemon2) + ") " + pokemon2[
        1])
    print("   " + pokemon1[1] + ": " + str(prediction_pokemon_1[0]))
    print("   " + pokemon2[1] + ": " + str(prediction_pokemon_2[0]))
    print("")
    if prediction_pokemon_1 > prediction_pokemon_2:
        print(pokemon1[1].upper() + " EST LE VAINQUEUR !")
    else:
        print(pokemon2[1].upper() + " EST LE VAINQUEUR !")


# Chargement du Pokedex et lancement d'un combat
with open("datas/pokedex.csv", newline='', encoding="ISO-8859-1") as csvfile:
    pokedex_in = csv.reader(csvfile)
    next(pokedex_in)
    # mettre par ordre croissant
    prediction(500, 570, pokedex_in)
