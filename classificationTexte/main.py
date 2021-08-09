import re

import pandas as pnd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
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
# CountVectorizer --> matrice des occurences des différents mots dans les différentes phrases.
# TF-IDF --> faible si le mot présent dans beaucoup de phrase
# TF-IDF --> faible si le mot peu présent dans la phrase
# TF-IDF --> fort si le mot peu présent dans la phrase et dans beaucoup de message
etapes_apprentissage = Pipeline([('frequence', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('algorithme', MultinomialNB())])

#######################################
# ------ Phase d'apprentissage ------ #
#######################################
print("Phase d'apprentissage")
modele = etapes_apprentissage.fit(X_train, y_train)
print(classification_report(y_test, modele.predict(X_test), digits=4))
#               precision    recall  f1-score   support
#            0     0.8171    0.2900    0.4281       231
#            1     0.7851    0.9756    0.8700       614
#     accuracy                         0.7882       845
#    macro avg     0.8011    0.6328    0.6491       845
# weighted avg     0.7938    0.7882    0.7492       845
# Précision de la classification de 79%


#################################
# ------ Nouvelle Phrase ------ #
#################################
phrase = "Why should trust scientists with global warming if they didnt know Pluto wasnt a planet"
print(phrase)

# 1 Normalisation
phrase = normalisation(phrase)

# 2 Suppression des stops words
phrase = ' '.join([mot for mot in phrase.split() if mot not in stopWords])

# 3 Stemmatization
phrase = ' '.join([stemmer.stem(mot) for mot in phrase.split(' ')])

# 4 Lemmitization
phrase = ' '.join([lemmatizer.lemmatize(mot) for mot in phrase.split(' ')])
print(phrase)

# 5 prédiction
prediction = modele.predict([phrase])
print(prediction[0])
if prediction[0] == 0:
    print(">> Ne croit pas au rechauffement climatique...")
else:
    print(">> Croit au rechauffement climatique...")

####################################
# ------ Utilisation de SVM ------ #
####################################
# 1 Pipeline
etapes_apprentissage = Pipeline([('frequence', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('algorithme', svm.SVC(kernel='linear', C=2))])

# 2 Apprentissage
modele_svms = etapes_apprentissage.fit(X_train, y_train)

print(classification_report(y_test, modele_svms.predict(X_test), digits=4))
#               precision    recall  f1-score   support
#            0     0.7043    0.6150    0.6566       213
#            1     0.8756    0.9130    0.8939       632
#     accuracy                         0.8379       845
#    macro avg     0.7899    0.7640    0.7753       845
# weighted avg     0.8324    0.8379    0.8341       845
# Précision de la classification de 83%

# 3 Recherche du meilleur paramètre C
parametresC = {'algorithme__C': (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12)}

rechercheCOptimal = GridSearchCV(etapes_apprentissage, parametresC, cv=2)
rechercheCOptimal.fit(X_train, y_train)
print(rechercheCOptimal.best_params_)

# 4 utilisation du nouveau Paramètre C=1
etapes_apprentissage = Pipeline([('frequence', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('algorithme', svm.SVC(kernel='linear', C=1))])

modele = etapes_apprentissage.fit(X_train, y_train)
print(classification_report(y_test, modele.predict(X_test), digits=4))

#               precision    recall  f1-score   support
#            0     0.7151    0.5775    0.6390       213
#            1     0.8663    0.9225    0.8935       632
#     accuracy                         0.8355       845
#    macro avg     0.7907    0.7500    0.7662       845
# weighted avg     0.8282    0.8355    0.8293       845
# Précision de la classification de 82%
