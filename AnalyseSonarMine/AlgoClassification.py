# Algorithme de "boosting"
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# fonction de calcul du précision accuracy_score pour la classification
from sklearn.metrics import accuracy_score
# Pour tester tous les hyperparamtres du SVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class AlgoClassification:

    @staticmethod
    def algo_classification(X_APPRENTISSAGE, X_VALIDATION, Y_APPRENTISSAGE, Y_VALIDATION, c_opt):
        # REGRESSION LOGISTIQUE
        regression_logistique = LogisticRegression()
        regression_logistique.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
        predictions = regression_logistique.predict(X_VALIDATION)
        print("Regression logistique: " + str(accuracy_score(predictions, Y_VALIDATION)))
        # ARBRE DE DECISION
        arbre_decision = DecisionTreeClassifier()
        arbre_decision.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
        predictions = arbre_decision.predict(X_VALIDATION)
        print("Arbre de décision:  " + str(accuracy_score(predictions, Y_VALIDATION)))
        # FORET ALEATOIRES
        foret_aleatoire = RandomForestClassifier()
        foret_aleatoire.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
        predictions = foret_aleatoire.predict(X_VALIDATION)
        print("Foret aléatoire: " + str(accuracy_score(predictions, Y_VALIDATION)))
        # K PLUS PROCHES VOISINS
        knn = KNeighborsClassifier()
        knn.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
        predictions = knn.predict(X_VALIDATION)
        print("K plus proche voisins: " + str(accuracy_score(predictions, Y_VALIDATION)))
        # MACHINE VECTEURS DE SUPPORT
        svm = SVC(gamma='auto')
        svm.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
        predictions = svm.predict(X_VALIDATION)
        print("Machine vecteurs de support: " + str(accuracy_score(predictions, Y_VALIDATION)))

        # Test du gradient boosting pour voir s'il est le meilleur
        gradient_boosting = GradientBoostingClassifier()
        gradient_boosting.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
        predictions = svm.predict(X_VALIDATION)
        print("GRADIENT BOOSTING: " + str(accuracy_score(predictions, Y_VALIDATION)))

        # MACHINE A VECTEURS DE SUPPORT OPTIMISE
        svm = SVC(C=c_opt, gamma='auto')
        svm.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
        predictions = svm.predict(X_VALIDATION)
        print("Machine à vecteurs de support optimisé: " + str(accuracy_score(predictions, Y_VALIDATION)))
