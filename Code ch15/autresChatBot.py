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
#
#  Code issu de la documentation du module NLTK : 
#          https://www.nltk.org/_modules/nltk/chat.html
#  
#-----------------------------------------------------------------------------------------


from __future__ import print_function

from nltk.chat.util import Chat
from nltk.chat.eliza import eliza_chat
from nltk.chat.iesha import iesha_chat
from nltk.chat.rude import rude_chat
from nltk.chat.suntsu import suntsu_chat
from nltk.chat.zen import zen_chat

bots = [
    (eliza_chat, 'Eliza (Pyschiatre)'),
    (iesha_chat, 'Iesha (Adolescent junky)'),
    (rude_chat, 'Rude (ChatBot abusif)'),
    (suntsu_chat, 'Suntsu (Proverbes chinois)'),
    (zen_chat, 'Zen (Perles de sagesses)'),
]


def chatbots():
    import sys

    print('Quel chatBot souhaitez vous tester ?')
    botcount = len(bots)
    for i in range(botcount):
        print('  %d: %s' % (i + 1, bots[i][1]))
    while True:
        print('\nChoisissez votre chatBot 1-%d: ' % botcount, end=' ')
        choice = sys.stdin.readline().strip()
        if choice.isdigit() and (int(choice) - 1) in range(botcount):
            break
        else:
            print('   Erreur: ce chatBot n\'existe pas')

    chatbot = bots[int(choice) - 1][0]
    chatbot()

chatbots()