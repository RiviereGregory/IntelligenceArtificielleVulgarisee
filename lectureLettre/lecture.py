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

    (grabbed, frame) = webCam.read()
    # image en HSV
    tsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # transformation en gris
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # activation contour avec canny
    contours_canny = cv2.Canny(gris, 30, 200)

    contours, hierarchy = cv2.findContours(contours_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        perimetre = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.012 * perimetre, True)
        x, y, w, h = cv2.boundingRect(approx)

        # On encadre la zone d'écriture en fonction des paramètres de longueur et largeur de l'ardoise
        if len(approx) == 4:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Affichage de l'image capturée par la webcam
    cv2.imshow("IMAGE", frame)
    cv2.imshow("HSV", tsv)
    cv2.imshow("GRIS", gris)
    cv2.imshow("CANNY", contours_canny)

    # Condition de sortie de la boucle While
    # > Touche Escape pour quitter
    key = cv2.waitKey(1)
    if key == 27:
        break

# On libère la webCam et on détruit toutes les fenêtres
webCam.release()
cv2.destroyAllWindows()
