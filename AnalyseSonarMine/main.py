import pandas as pnd

# ------- Préparation des données -------- #
observations = pnd.read_csv("datas/sonar.all-data.csv")
print(observations.columns.values)
# ['0.0200' '0.0371' '0.0428' '0.0207' '0.0954' '0.0986' '0.1539' '0.1601'
#  '0.3109' '0.2111' '0.1609' '0.1582' '0.2238' '0.0645' '0.0660' '0.2273'
#  '0.3100' '0.2999' '0.5078' '0.4797' '0.5783' '0.5071' '0.4328' '0.5550'
#  '0.6711' '0.6415' '0.7104' '0.8080' '0.6791' '0.3857' '0.1307' '0.2604'
#  '0.5121' '0.7547' '0.8537' '0.8507' '0.6692' '0.6097' '0.4943' '0.2744'
#  '0.0510' '0.2834' '0.2825' '0.4256' '0.2641' '0.1386' '0.1051' '0.1343'
#  '0.0383' '0.0324' '0.0232' '0.0027' '0.0065' '0.0159' '0.0072' '0.0167'
#  '0.0180' '0.0084' '0.0090' '0.0032' 'R']

print(observations.shape)
# (207, 61) --> 208 observations et 62 features

# Ajout d'un nom pour chaque feature
observations = pnd.read_csv("datas/sonar.all-data.csv", names=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
                                                               "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17",
                                                               "F18", "F19",
                                                               "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27",
                                                               "F28", "F29",
                                                               "F30", "F31", "F32", "F33", "F34", "F35", "F36", "F37",
                                                               "F38", "F39",
                                                               "F40", "F41", "F42", "F43", "F44", "F45", "F46", "F47",
                                                               "F48", "F49",
                                                               "F50", "F51", "F52", "F53", "F54", "F55", "F56", "F57",
                                                               "F58", "F59",
                                                               "F60", "OBJET"])

# Desactivation du nombre maximum de colonnes du DataFrame à afficher
pnd.set_option('display.max_columns', None)

# Affichage des 10 premieres observations
print(observations.head(10))

# Transformation de la caractéristique Objet
# 1 pour type mine
# 0 pour type rocher
observations['OBJET'] = (observations['OBJET'] == 'M').astype(int)

# Vérification qu'il ne manque pas d'information
print(observations.info())
# --> manque aucune valeurs


# ------- Analyse données -------- #
