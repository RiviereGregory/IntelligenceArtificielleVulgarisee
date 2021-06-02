#-------------------------------------------------------
# @Auteur : Aurélien Vannieuwenhuyze
# @Entreprise : Junior Makers Place
# @Livre
# @Chapitre : 03 - Les fondamentaux du langage python
#
# Modules necessaires : --
#-------------------------------------------------------



#----------------------------------------------
# FONCTIONS
#----------------------------------------------

def calculDeLaSurfaceANettoyer(listeDeZones):
    surfaceANettoyer = 0
    for zone in listeDeZones:
        longueur = zone.get("longueur")/100
        largeur = zone.get("largeur")/100
        calcul = longueur*largeur
        print (str(longueur)+" x "+str(largeur)+"= "+str(calcul))
        surfaceANettoyer = surfaceANettoyer +calcul
    return (surfaceANettoyer)


def tempsNetoyageEnMinutes(surfaceANettoyer, tempsPourUnMetreCarre):
    return round(surfaceANettoyer*tempsPourUnMetreCarre)

#----------------------------------------------
# APPLICATION
#----------------------------------------------

#Utilisation d'un tuple pour le parametrage de l'application
#Nom du robot, temps en minutes pour nettoyer un metre carré
parametres = ("robot_aspiro",2)

# Utilisation de dictionnaires pour créer les zones
zone1={"longueur":500,"largeur":150}
zone2={"longueur":309,"largeur":480}
zone3={"longueur":101,"largeur":480}
zone4={"longueur":90,"largeur":220}

# Utilisation d'une liste permettant de stocker de nos différentes zones
zones = []
zones.append(zone1)
zones.append(zone2)
zones.append(zone3)
zones.append(zone4)

#Appel de la fonction permettant de calculer la surface à nettoyer
surfaceANettoyer = calculDeLaSurfaceANettoyer(zones)
print("La surface total à nettoyer est de : "+str(surfaceANettoyer)+ " m2")

#Appel de la fonction permettant de déterminer le temps de nettoyage
tempsEstime = tempsNetoyageEnMinutes(surfaceANettoyer,parametres[1])
print("Le temps estimé est de: "+str(tempsEstime)+" minutes")

#Ajout d'une condition se déclenchant en fonction du temps de nettoyage
if tempsEstime > 55:
    print(parametres[0]+" dit : Je pense que cela va prendre un peu de temps !")