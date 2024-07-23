import cv2
import numpy as np
import os

# Aktuelles Verzeichnis
project_dir = os.path.dirname(os.path.abspath(__file__))

print(project_dir)

# Pfade der Bilder
in_folder = os.path.join(project_dir, 'imgs')
out_folder = os.path.join(project_dir, 'imgs_edited')
calib_file = os.path.join(out_folder, 'calib_data.npz')

# Output-Folder erstellen (falls nicht vorhanden)
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

if not os.path.exists(calib_file):
    # -----------------------------------
    # (*) LADEN UND BEARBEITEN DER KALIBIERUNGSBILDER
    # -----------------------------------
    # Bearbeiten der Bilder
    def better_img_edit(img):
        # Bild in Graustufen umwandeln
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    # Bilder bearbeiten und speichern
    for fname in os.listdir(in_folder):
        if fname.endswith(".jpg"):
            img_path = os.path.join(in_folder, fname)
            img = cv2.imread(img_path)
            edited_img = better_img_edit(img)
            edited_img_path = os.path.join(out_folder, fname)
            cv2.imwrite(edited_img_path, edited_img)

    # -----------------------------------
    # (1) KALIBRIERUNG
    # -----------------------------------
    # Schachbrettgröße (innere Ecken)
    chessboard_size = (8, 4)

    # Größe eines Quadrats in Millimetern
    square_size = 28.57

    # Kriterien für die Ecksuche (max. Iterationen + Genauigkeit)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D Punkte in der realen Welt
    obj_point = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    obj_point *= square_size

    # Arrays für die Speicherung der 3D Punkte und der 2D Bildpunkte für alle eingelesenen Bilder
    obj_points = []  # 3D Punkte in der realen Welt
    img_points = []  # 2D Punkte in den Bildern
    img_names = []   # Liste um Namen zu speichern für RMS (Bild <-> RMS Zuweisung)

    # Punktsuche (aus dem editierten Bilder-Ordner)
    for fname in os.listdir(out_folder):
        if fname.endswith(".jpg"):
            img_path = os.path.join(out_folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Irgendein Fehler beim Laden des Bildes")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                obj_points.append(obj_point)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners2)
                img_names.append(fname)
                cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                cv2.imshow('img', img)
                cv2.waitKey(50)

    cv2.destroyAllWindows()

    # Kamerakalibrierung (mit den Objekt- und Bildpunkten)
    ret, cam_matrix, distortion_coeff, rotation_vectors, translation_vectors = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Kalibrierungsergebnisse speichern
    np.savez(calib_file,
             cam_matrix=cam_matrix,
             distortion_coeff=distortion_coeff,
             rotation_vectors=rotation_vectors,
             translation_vectors=translation_vectors)
    
    # -----------------------------------
    # (3) REPROJECTIONSFEHLER
    # -----------------------------------

    # Berechnung des Reprojektionsfehlers
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rotation_vectors[i], translation_vectors[i], cam_matrix, distortion_coeff)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error /= len(obj_points)
    print(f"Gesamter Reprojektionsfehler: {mean_error}")
else:
    # Kalibrierungsergebnisse laden
    calib_data = np.load(calib_file)
    cam_matrix = calib_data['cam_matrix']
    distortion_coeff = calib_data['distortion_coeff']
    rotation_vectors = calib_data['rotation_vectors']
    translation_vectors = calib_data['translation_vectors']

# Ausgabe Kalibrierungsergebnisse
"""
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
"""

# -----------------------------------
# (2) ENTZERRUNG EINES EINGABE BILDES
# -----------------------------------

image_name = "240500013_markings_rotated.png" # 240500013_markings_rotated.png
img_path = os.path.join(in_folder, image_name)
print(img_path)

img = cv2.imread(img_path)

h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, distortion_coeff, (w, h), 1, (w, h))

# Verzerrung rausnehmen
dst = cv2.undistort(img, cam_matrix, distortion_coeff, None, newcameramtx)

# Bild zuschneiden
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

# Bild speichern
undistorted_img_path = os.path.join(out_folder, 'entzerrt_' + image_name)
cv2.imwrite(undistorted_img_path, dst)

# Entzerrtes Bild anzeigen
cv2.namedWindow('entzerrt_img', cv2.WINDOW_NORMAL)
cv2.imshow('entzerrt_img', dst)
cv2.waitKey(100)
cv2.destroyAllWindows()


# AUF BASIS VON: https://stackoverflow.com/questions/51272055/opencv-unproject-2d-points-to-3d-with-known-depth-z
def project(points, intrinsic, distortion):
    rvec = tvec = np.array([0.0, 0.0, 0.0])
    projected_points, _ = cv2.projectPoints(points, rvec, tvec, intrinsic, distortion)

    return np.squeeze(projected_points, axis=1)

def unproject(points, Z, intrinsic, distortion):
    f_x = intrinsic[0, 0]
    f_y = intrinsic[1, 1]
    c_x = intrinsic[0, 2]
    c_y = intrinsic[1, 2]

    # (1) Entzerren
    points_undistorted = cv2.undistortPoints(np.expand_dims(points.astype(np.float32), axis=1), intrinsic, distortion, P=intrinsic)
    points_undistorted = np.squeeze(points_undistorted, axis=1)

    # (2) )Reprojektion
    result = []

    for idx in range(points_undistorted.shape[0]):
        z = Z[0] if len(Z) == 1 else Z[idx]
        x = (points_undistorted[idx, 0] - c_x) / f_x * z
        y = (points_undistorted[idx, 1] - c_y) / f_y * z

        result.append([x, y, z])

    return np.array(result)

# Kalibrierungsergebnisse aus der Datei laden
#calib_data = np.load(calib_file)
#cam_matrix = calib_data['cam_matrix']
#distortion_coeff = calib_data['distortion_coeff']

# Punktkoordinaten und Tiefe (Z)
#u, v = 2090, 1940 
#depth = 171  # Tiefe in mm

# Punkt und Tiefe in Projektions- und Umprojektionsfunktionen einfügen
#point_single = np.array([[u, v]], dtype=np.float32)
#Z = np.array([depth], dtype=np.float32)

# Umprojektion des Punkts von 2D -> 3D
#point_single_unprojected = unproject(point_single, Z, cam_matrix, distortion_coeff)

#print("Erwarteter Punkt:", [u, v, depth])
#print("Berechneter Punkt:", point_single_unprojected[0])

# Einzelfunktion für Punktprojektion
def get_single_projection(u, v, depth):
    point = np.array([[u, v]], dtype=np.float32)
    Z = np.array([depth], dtype=np.float32)
    point_unprojected = unproject(point, Z, cam_matrix, distortion_coeff)

    print("Erwarteter Punkt:", [u, v, depth])
    print("Berechneter Punkt:", point_unprojected[0])

    return point_unprojected[0]


# Beispielaufruf für die Einzelfunktion get_single_projection u, v, depth (z)
u, v, depth = 2090, 1940, 171
# Ausgabe bereits hier
single_point_result = get_single_projection(u, v, depth)

# --------------------------------------------------------------

# Funktion für Liste von Punkten u,v
def get_multiple_projections(points_with_depth):
    results = []

    for (u, v, depth) in points_with_depth:
        point_unprojected = get_single_projection(u, v, depth)
        results.append(point_unprojected)

    return results



