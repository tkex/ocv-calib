import cv2
import os
import numpy as np

# Aktuelles Verzeichnis
project_dir = os.path.dirname(os.path.abspath(__file__))

# Pfade der Bilder
in_folder = os.path.join(project_dir, 'imgs')
out_folder = os.path.join(project_dir, 'imgs_edited')

# Output-Folder erstellen (falls nicht vorhanden)
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# Bearbeiten der Bilder
def better_img_edit(img):

    # Bild in Graustufen umwandeln
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kontrast erhöhen
    # edited_img = cv2.equalizeHist(gray)

    return gray

# Bilder bearbeiten und speichern
for fname in os.listdir(in_folder):

    if fname.endswith(".jpg"):

        img_path = os.path.join(in_folder, fname)
        img = cv2.imread(img_path)

        edited_img = better_img_edit(img)
        edited_img_path = os.path.join(out_folder, fname)

        cv2.imwrite(edited_img_path, edited_img)



# Schachbrettgröße (innere Ecken)
chessboard_size = (8, 4)

# Größe eines Quadrats in Millimetern
square_size = 28.77

# Kriterien für die Ecksuche (max. Iterationen + Genauigkeit)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D Punkte in der realen Welt
# (Array 3D Punkte welche die Position der Schachbrettecken in der echten Welt darstellen. 32 Punkte mit Z null.)
obj_point = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_point[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Skalierung der 3D Punkte mit der tatsächlichen Größe der Quadrate
obj_point = obj_point * square_size

# Arrays für die Speicherung der 3D Punkte und der 2D Bildpunkte für alle  eingelesenen Bilder
# dh. die Position der Schachbrettecken in der realen Welt (z.B. 0,0,0; 1,0,0; 2,0,0; ..., 7,3,0)

# 3D Punkte in der realen Welt
obj_points = []

# 2D Punkte in den Bildern
img_points = [] 

# Liste um Namen zu spechen für RMS (Bild <-> RMS Zuweisung)
img_names = []

# Punktsuche (aus dem editierten Bilder-Ordner)
for fname in os.listdir(out_folder):

    if fname.endswith(".jpg"):

        img_path = os.path.join(out_folder, fname)

        # Einlesen des Bilds
        img = cv2.imread(img_path)

        print(f"Eingelesenes Bild: {fname}")

        # Präventiv, passiert aber in der Edit-Funktion bereits
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Finden der Schachbrettecken
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # Wenn Ecken gefunden wurden -> Objektpunkte und Bildpunkte speichern
        if ret:
            obj_points.append(obj_point)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

            # Bild in Namensliste hinzufügen (für RMS Ermittlung)
            img_names.append(fname)

            # Einzeichnen der Ecken
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)

            # Anzeige des Bildes samt Ecke
            cv2.imshow('img', img)

            # Warte kurz
            cv2.waitKey(500)

cv2.destroyAllWindows()


# Kamerakalibrierung (mit den Objekt- und Bildpunkten)
# ret: der RMS (Root Mean Square) Reprojektion-Fehler. Niedriger Wert -> bessere/genauere Kalibrierung.
ret, cam_matrix, distortion_coeff, rotation_vectors, translation_vectors = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Ausgabe Kalibrierungsergebnisse
print("Kalibrierung wurde abgeschlossen!")

print("Erfolgsrate (RMS-Fehler):", ret)

print("Kameramatrix:")
print(cam_matrix)

print("Verzerrungskoeffizienten:")
print(distortion_coeff)

print("Rotationsvektoren:")
print(rotation_vectors)

print("Translationsvektoren:")
print(translation_vectors)

# Reprojection-Error berechnen
#mean_error = 0

#for i in range(len(obj_points)):
#    img_points2, _ = cv2.projectPoints(obj_points[i], rotation_vectors[i], translation_vectors[i], cam_matrix, distortion_coeff)

#    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)

#    mean_error += error

#print("Gesamter Reprojection-Error: {}".format(mean_error / len(obj_points)))


# Reprojection-Error für jedes Bild berechnen und anzeigen
mean_error = 0

# Liste für Fehler je Bild
errors_per_image = []

for i in range(len(obj_points)):

    img_points2, _ = cv2.projectPoints(obj_points[i], rotation_vectors[i], translation_vectors[i], cam_matrix, distortion_coeff)

    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    mean_error += error

    errors_per_image.append((img_names[i], error))

# Durchschnittlichen Reprojection-Error berechnen
mean_error /= len(obj_points)

# Ergebnisse anzeigen
print("Gesamter Reprojection-Error: {}".format(mean_error))

for fname, error in errors_per_image:
    print(f"Reprojection-Error für {fname}: {error}")

# Bilder mit höchsten Reprojection-Errors finden
def get_img_error(img):
    return img[1]

errors_per_image.sort(key=get_img_error, reverse=True)

# Top 5 Bilder mit den höchsten Reprojection-Error
print("\nBilder mit den höchsten Reprojection-Error (sortiert):")

for fname, error in errors_per_image[:5]: 
    print(f"{fname}: {error}")