import cv2

print('Initialisation de la webcam')
webCam = cv2.VideoCapture(0)
if webCam.isOpened():
    longueurWebcam = webCam.get(3)
    largeurWebcam = webCam.get(4)
    print('Résolution:' + str(longueurWebcam) + " X " + str(largeurWebcam))
else:
    print('ERREUR')

while True:

    # Capture de l'image dans la variable Frame
    # La variable lectureOK est égale à True si la fonction read() est opérationnelle
    (lectureOK, frame) = webCam.read()

    # Affichage de l'image capturée par la webcam
    cv2.imshow("IMAGE", frame)

    # Condition de sortie de la boucle While
    # > Touche Escape pour quitter
    key = cv2.waitKey(1)
    if key == 27:
        break

# On libère la webCam et on détruit toutes les fenêtres
webCam.release()
cv2.destroyAllWindows()
