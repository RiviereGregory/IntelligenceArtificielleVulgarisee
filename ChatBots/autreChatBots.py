from __future__ import print_function

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
