import re

import pandas as pnd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

messagesTwitter = pnd.read_csv("datas/rechauffementClimatique.csv", ";")

print(messagesTwitter.shape)
print(messagesTwitter.head(2))

messagesTwitter['CROYANCE'] = (messagesTwitter['CROYANCE'] == 'Yes').astype(int)
print(messagesTwitter.head(100))


################################
# ------ NORMALISATION ------ #
##############################
def normalisation(message):
    message = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', message)
    message = re.sub('@[^\s]+', 'USER', message)
    message = message.lower().replace("ё", "е")
    message = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', message)
    message = re.sub(' +', ' ', message)
    return message.strip()


print("# ------ NORMALISATION ------ #")
messagesTwitter["TWEET"] = messagesTwitter["TWEET"].apply(normalisation)
print(messagesTwitter.head(10))

#############################
# ------ STOP WORDS ------ #
###########################
# Suppression des mots les plus utilisés
print("# ------ STOP WORDS ------ #")
stopWords = stopwords.words('english')

messagesTwitter['TWEET'] = messagesTwitter['TWEET'].apply(
    lambda message: ' '.join([mot for mot in message.split() if mot not in stopWords]))
print(messagesTwitter.head(10))

#############################
# ---- Stemmatisation ---- #
###########################
print("# ---- Stemmatisation ---- #")
stemmer = SnowballStemmer('english')
messagesTwitter['TWEET'] = messagesTwitter['TWEET'].apply(
    lambda message: ' '.join([stemmer.stem(mot) for mot in message.split(' ')]))
print(messagesTwitter.head(10))

############################
# ---- Lemmatization ---- #
##########################
print("# ---- Lemmatization ---- #")

lemmatizer = WordNetLemmatizer()
messagesTwitter['TWEET'] = messagesTwitter['TWEET'].apply(
    lambda message: ' '.join([lemmatizer.lemmatize(mot) for mot in message.split(' ')]))
print(messagesTwitter.head(10))

print("Fin de la préparation !")

###################################
# Jeux d'apprentissage et de test #
###################################
X_train, X_test, y_train, y_test = train_test_split(messagesTwitter['TWEET'].values, messagesTwitter['CROYANCE'].values,
                                                    test_size=0.2)

########################################
# Creation du pipeline d'apprentissage #
########################################

etapes_apprentissage = Pipeline([('frequence', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('algorithme', MultinomialNB())])
