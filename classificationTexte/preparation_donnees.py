####################################
# ------ IMPORT DES MODULES ------ #
####################################

import chardet
import pandas as pnd

####################################
# --- PREPARATIONS DES DONNEES --- #
####################################
# Savoir quel est l'encodage du csv
with open("datas/rechauffementClimatique_non_preparees.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
print(result)
# {'encoding': 'Windows-1252', 'confidence': 0.73, 'language': ''}

# Chargement des données du rechauffement climantique
messagesTwitter = pnd.read_csv("datas/rechauffementClimatique_non_preparees.csv", ";", encoding="Windows-1252")

# Affichage des colonnes du dataFrame
print(messagesTwitter.columns.values)
# ['Column1' 'Column2' 'Column3']
# Nom des Colonnes pas pertinantes

# Affichage des 10 premières lignes
print(messagesTwitter.head(10))
#                                              Column1  ...               Column3
# 0                                              tweet  ...  existence.confidence
# 1  Global warming report urges governments to act...  ...                     1
# 2  Fighting poverty and global warming in Africa ...  ...                     1
# 3  Carbon offsets: How a Vatican forest failed to...  ...                0.8786
# 4  Carbon offsets: How a Vatican forest failed to...  ...                     1
# 5  URUGUAY: Tools Needed for Those Most Vulnerabl...  ...                0.8087
# 6  RT @sejorg: RT @JaymiHeimbuch: Ocean Saltiness...  ...                     1
# 7  Global warming evidence all around us|A messag...  ...                     1
# 8  Migratory Birds' New Climate Change Strategy: ...  ...                     1
# 9  Southern Africa: Competing for Limpopo Water: ...  ...                     1
# [10 rows x 3 columns]
# La première ligne semble être la valeur des colonnes

# Comptage du nombre d'observation et de features
print(messagesTwitter.shape)
# (6091, 3)

# Information sur le jeu de données (vérification que les données sont completes)
print(messagesTwitter.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 6091 entries, 0 to 6090
# Data columns (total 3 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   Column1  6091 non-null   object
#  1   Column2  4226 non-null   object
#  2   Column3  6088 non-null   object
# dtypes: object(3)
# memory usage: 142.9+ KB
# None
# on voit que l'on a des valeurs null pour certaine lignes

# suppression des lignes avec valeurs manquantes
index_with_nan = messagesTwitter.index[messagesTwitter.isnull().any(axis=1)]
print(index_with_nan)
# Int64Index([  15,   35,   41,   48,   49,   51,   52,   53,   54,   55,
#             ...
#             6058, 6061, 6066, 6067, 6068, 6070, 6073, 6075, 6084, 6087],
#            dtype='int64', length=1865)
messagesTwitter.drop(index_with_nan, 0, inplace=True)
# vérification
print(messagesTwitter.info())
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 4226 entries, 0 to 6090
# Data columns (total 3 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   Column1  4226 non-null   object
#  1   Column2  4226 non-null   object
#  2   Column3  4226 non-null   object
# dtypes: object(3)
# memory usage: 132.1+ KB
# None

# suppression de la première ligne (le nom des colonnes)
messagesTwitter.to_csv("datas/rechauffementClimatique_preparees.csv", ";", header=False, index=False)

# Modification noms de colonnes
messagesTwitterP = pnd.read_csv("datas/rechauffementClimatique_preparees.csv", ";")
messagesTwitterP.columns = ['TWEET', 'CROYANCE', 'CONFIENCE']
# Enregistremetn nouveau fichier avec bon nom des colonnes
messagesTwitterP.to_csv("datas/rechauffementClimatique_preparees.csv", ";", index=False)
# vérification Globale
print(messagesTwitterP.columns.values)
# ['TWEET' 'CROYANCE' 'CONFIENCE']
print(messagesTwitterP.head(10))
#                                                TWEET CROYANCE  CONFIENCE
# 0  Global warming report urges governments to act...      Yes     1.0000
# 1  Fighting poverty and global warming in Africa ...      Yes     1.0000
# 2  Carbon offsets: How a Vatican forest failed to...      Yes     0.8786
# 3  Carbon offsets: How a Vatican forest failed to...      Yes     1.0000
# 4  URUGUAY: Tools Needed for Those Most Vulnerabl...      Yes     0.8087
# 5  RT @sejorg: RT @JaymiHeimbuch: Ocean Saltiness...      Yes     1.0000
# 6  Global warming evidence all around us|A messag...      Yes     1.0000
# 7  Migratory Birds' New Climate Change Strategy: ...      Yes     1.0000
# 8  Southern Africa: Competing for Limpopo Water: ...      Yes     1.0000
# 9  Global warming to impact wheat, rice productio...      Yes     1.0000
print(messagesTwitterP.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4225 entries, 0 to 4224
# Data columns (total 3 columns):
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   TWEET      4225 non-null   object
#  1   CROYANCE   4225 non-null   object
#  2   CONFIENCE  4225 non-null   float64
# dtypes: float64(1), object(2)
# memory usage: 99.1+ KB
# None
