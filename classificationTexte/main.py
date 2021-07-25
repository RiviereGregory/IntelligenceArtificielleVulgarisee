import re

import pandas as pnd
from nltk.corpus import stopwords

messagesTwitter = pnd.read_csv("datas/rechauffementClimatique.csv", ";")

print(messagesTwitter.shape)
print(messagesTwitter.head(2))

messagesTwitter['CROYANCE'] = (messagesTwitter['CROYANCE'] == 'Yes').astype(int)
print(messagesTwitter.head(100))


################################
# ------ NORMALISATION ------ #
#############################
def normalisation(message):
    message = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', message)
    message = re.sub('@[^\s]+', 'USER', message)
    message = message.lower().replace("ё", "е")
    message = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', message)
    message = re.sub(' +', ' ', message)
    return message.strip()


messagesTwitter["TWEET"] = messagesTwitter["TWEET"].apply(normalisation)
print(messagesTwitter.head(10))

#############################
# ------ STOP WORDS ------ #
##########################
# Suppression des mots les plus utilisés
stopWords = stopwords.words('english')

messagesTwitter['TWEET'] = messagesTwitter['TWEET'].apply(
    lambda message: ' '.join([mot for mot in message.split() if mot not in stopWords]))
print(messagesTwitter.head(10))

