#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/


#Module de lecture de fichiers CSV
import csv

#Module de chargement du modèle d'apprentissage
from sklearn.externals import joblib

def rechercheInformationsPokemon(numPokemon,Pokedex):
    infosPokemon = []
    for pokemon in Pokedex:
        if (int(pokemon[0])==numPokemon):
            infosPokemon = [pokemon[0],pokemon[1],pokemon[4],pokemon[5],pokemon[6],pokemon[7],pokemon[8],pokemon[9],pokemon[10]]
            break
    return infosPokemon


def prediction (numeroPokemon1, numeroPokemon2,Pokedex):
    pokemon1 = rechercheInformationsPokemon(numeroPokemon1, Pokedex)
    pokemon2 = rechercheInformationsPokemon(numeroPokemon2, Pokedex)
    modele_prediction = joblib.load('modele/modele_pokemon.mod')
    prediction_Pokemon_1 = modele_prediction.predict([[pokemon1[2],pokemon1[3],pokemon1[4],pokemon1[5],pokemon1[6],pokemon1[7],pokemon1[8]]])
    prediction_Pokemon_2 = modele_prediction.predict([[pokemon2[2], pokemon2[3], pokemon2[4], pokemon2[5], pokemon2[6], pokemon2[7], pokemon2[8]]])
    print ("COMBAT OPPOSANT : ("+str(numeroPokemon1)+") "+pokemon1[1]+" à ("+str(numeroPokemon2)+") "+pokemon2[1])
    print ("   "+pokemon1[1]+": "+str(prediction_Pokemon_1[0]))
    print("   " +pokemon2[1] + ": " + str(prediction_Pokemon_2[0]))
    print ("")
    if prediction_Pokemon_1>prediction_Pokemon_2:
        print(pokemon1[1].upper()+" EST LE VAINQUEUR !")
    else:
        print(pokemon2[1].upper() + " EST LE VAINQUEUR !")



#Chargement du Pokedex et lancement d'un combat
with open("datas/pokedex_application.csv", newline='') as csvfile:
    pokedex = csv.reader(csvfile)
    next(pokedex)
    prediction(368,598,pokedex)

