#-----------------------------------------------------------------------------------------
# @Auteur : Aurélien Vannieuwenhuyze
# @Entreprise : Junior Makers Place
# @Livre :
# @Chapitre : 15 - Hommage au premier ChatBot !
#
# Modules necessaires :
#   NLTK 3.4.1
#
# Pour installer un module :
#   Cliquer sur le menu File > Settings > Project:nom_du_projet > Project interpreter > bouton +
#   Dans la zone de recherche en haut à gauche saisir le nom du module
#   Choisir la version en bas à droite
#   Cliquer sur le bouton install situé en bas à gauche
#-----------------------------------------------------------------------------------------




from __future__ import print_function

#Le sous module reflection permet de faire les liens entre les clés et les réponses à apporter
from nltk.chat.util import Chat, reflections


Cles_valeurs = (
        (
            r'Bonjour(.*)',
            (
                "Bonjour... je suis contente de discuter avec toi aujourd'hui",
                "Salut !! Quoi de neuf aujourd'hui ?",
            ),
        ),
        (
            r'J\'ai besoin (.*)',
            (
                    "Pourquoi as-tu besoin  %1 ?",
                    "Est-ce que ça t'aiderait vraiment  %1 ?",
                    "Es tu sûr d'avoir besoin  %1 ?",
            ),
        ),
        (
            r'Pourquoi ne pas (.*)',
            (
                "Tu crois vraiment que je n'ai pas 1 % ?",
                "Peut-être qu'un jour, je finirai par %1."
                "Tu veux vraiment que je fasse 1 % ?",
            ),
        ),
        (
            r'Pourquoi je ne peux pas (.*)',
            (
                "Penses tu que tu devrais être capable de %1 ?",
                "Si tu pouvais %1, que ferais-tu ?",
                "Je ne sais pas... pourquoi tu ne peux pas %1 ?",
                "Tu as vraiment essayé ?",
            ),
        ),
        (
            r'Je ne peux pas (.*)',
                (
                "Comment sais tu que tu ne peux pas %1 ?",
                "Tu pourrais peut-être faire 1 % si tu essaies."
                "Qu'est-ce qu'il te faudrait pour avoir 1 % ?",
            ),
        ),
        (
            r'Je suis (.*)',
            (
                "Es-tu venu me voir parce que tu es %1 ?",
                "Depuis combien de temps êtes-vous %1 ?",
                "Que penses-tu d'être %1 ?",
                "Qu'est-ce que ça te fait d'être %1 ?",
                "Aimes tu être %1 ?",
                "Pourquoi me dis-tu que tu es à 1 % ?",
                "Pourquoi penses-tu que tu es à 1 % ?",
            ),
        ),

        (
            r'Es-tu (.*)',
            (
                "Pourquoi est-ce important que je sois %1 ?",
                "Tu préférerais que je ne sois pas %1 ?",
                "Tu crois peut-être que je suis %1."
                "Je suis peut-être %1 -- qu'en penses-tu ?",
            ),
        ),
        (
            r'Quoi (.*)',
            (
                "Pourquoi cette question ?",
                "En quoi une réponse à ça t'aiderait ?",
                "Qu'en penses-tu ?",
            ),
        ),
        (
            r'Comment (.*)',
            (
                "Comment tu crois ?",
                "Tu peux peut-être répondre à ta propre question."
                "Qu'est-ce que tu demandes vraiment ?",
            ),
        ),
        (
            r'Parce que (.*)',
            (
                "C'est la vraie raison ?",
                "Quelles autres raisons me viennent à l'esprit ?",
                "Cette raison s'applique-t-elle à autre chose ?",
                "Si %1, quoi d'autre doit être vrai ?",
            ),
        ),
        (
            r'(.*) désolé (.*)',
            (
                "Il y a de nombreuses fois où il n'est pas nécessaire de s'excuser."
                "Qu'est-ce que tu ressens quand tu t'excuses ?",
            ),
        ),

        (
            r'Je pense que (.*)',
            ("Doute tu de %1 ?",
             "Tu le penses vraiment ?",
             "Mais tu n'es pas sûr de %1 ?"),
        ),

        (
            r'Oui',
                 ('Tu me sembles bien sûr.',
                  "OK, mais peux-tu développer un peu ?")
        ),
        (
            r'(.*) ordinateur(.*)',
            (
                "Tu parles vraiment de moi ?",
                "Ça te paraît étrange de parler à un ordinateur ?",
                "Comment te sens tu avec les ordinateurs ?",
                "Te sens tu menacé par les ordinateurs ?",
            ),
        ),
        (
            r'Est-ce (.*)',
            (
                "Penses-tu que c'est %1 ?",
                "Peut-être que c'est %1 -- qu'en penses-tu ?",
                "Si c'était %1, que ferais-tu ?",
                "Ça pourrait bien être ce %1.",
            ),
        ),
        (
            r'C\'est (.*)',
            (
            "Tu me sembles très certain.",
            "Si je te disais que ce n'est probablement pas %1, que ressentirais-tu ?",
            ),
        ),
        (
            r'Peux-tu (.*)',
            (
                "Qu'est-ce qui te fait croire que je ne peux pas faire 1 % ?",
                "Si je pouvais %1, alors quoi ?",
                "Pourquoi me demandes-tu si je peux %1 ?",
            ),
        ),
        (
            r'Je peux (.*)',
            (
                "Peut-être que tu ne voulais pas de %1.",
                "Veux-tu être capable de %1 ?",
                    "Si tu pouvais %1, tu le ferais ?",
            ),
        ),
        (
            r'Vous êtes (.*)',
            (
                "Pourquoi penses tu que je suis %1 ?",
                "Est-ce que ça te fait plaisir de penser que je suis %1 ?",
                "Peut-être voudrais-tu que je sois %1.",
                "Tu parles peut-être vraiment de toi ?",
                "Pourquoi dis tu que je suis %1 ?",
                "Pourquoi penses tu que je suis %1 ?",
                "On parle de toi ou de moi ?",
            ),
        ),

    (
        r'Au revoir',
        (
            "Merci de m'avoir parlé.",
            "Au revoir."
        ),
    ),
    (
        r'(.*)',
        (
            "S'il te plaît, dis-m'en plus.",
            "Changeons un peu de sujet.... Parle-moi de toi.",
            "Peux tu m'en dire plus à ce sujet ?",
            "Pourquoi dis-tu ça ?",
            "Je vois.",
            "Très intéressant.",
            "Je vois.  dis m'en plus ?",
        ),
    ),
)


#Lancement du programme

eliza_chatbot = Chat(Cles_valeurs, reflections)

print("Programme Eliza\n---------")
print('=' * 72)
print("Bonjour.  Comment vas tu?")

eliza_chatbot.converse()
