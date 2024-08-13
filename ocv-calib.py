import cv2
import numpy as np
import os
import plotly.graph_objects as go

# Konstanten
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_FOLDER = os.path.join(PROJECT_DIR, 'imgs')
OUT_FOLDER = os.path.join(PROJECT_DIR, 'imgs_edited')
CALIB_FILE = os.path.join(OUT_FOLDER, 'calibration_data.npz')
CHESSBOARD_SIZE = (8, 4)
SQUARE_SIZE = 28.57
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
Z = 171  # Tiefe für die Umprojektion ; 2600 für Test 2: Realmessung.

class ImageProcessor:
    """
    Bearbeiten und Speichern von Bildern, um Bilder zu drehen, in Graustufen umwandeln und die bearbeiteten Bilder im angegebenen Ausgabeordner (imgs_edited) zu speichern.
    """
    def __init__(self, in_folder, out_folder):
        # Initialisierung der Eingabe- und Ausgabeverzeichnisse
        self.in_folder = in_folder
        self.out_folder = out_folder

    def better_img_edit(self, img):
        # Bild drehen (90 Grad gegen den Uhrzeigersinn)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Bild in Graustufen umwandeln
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def process_images(self):
        # Durchlaufe alle Dateien im imgs_edited Ordner
        for fname in os.listdir(self.in_folder):
            # Falls Datei eine .jpg-Datei ist
            if fname.endswith(".jpg"):
                # Pfad erstellen (Input)
                img_path = os.path.join(self.in_folder, fname)
                # Bild einlesen
                img = cv2.imread(img_path)
                # Bild bearbeiten
                edited_img = self.better_img_edit(img)
                # Pfad erstellen
                edited_img_path = os.path.join(self.out_folder, fname)
                # Bearbeitetes Bild speichern
                cv2.imwrite(edited_img_path, edited_img)


class CameraCalibration:
    """
    Führt die Kalibrierung einer Kamera durch und verwendet Bilder, um Objektpunkte (3D in der realen Welt) und Bildpunkte (2D in den Bildern) zu finden. 
    Die Kalibrierungsergebnisse (Kameramatrix, Verzerrungskoeffizienten, Rotations- und Translationsvektoren) werden gespeichert und können geladen werden.
    Reprojektionsfehler wird berechnet, um die Genauigkeit der Kalibrierung zu überprüfen.
    """
    def __init__(self, out_folder, calib_file):
        self.out_folder = out_folder
        self.calib_file = calib_file
        self.obj_points = []  # 3D Punkte in der realen Welt
        self.img_points = []  # 2D Punkte in den Bildern
        self.img_names = []   # Liste um Namen zu speichern für RMS (Bild <-> RMS Zuweisung)
        self.cam_matrix = None
        self.distortion_coeff = None
        self.rotation_vectors = None
        self.translation_vectors = None
        self.ret = None

    def create_object_points(self):
        obj_point = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        obj_point[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        obj_point *= SQUARE_SIZE
        return obj_point

    def find_corners(self, gray):
        return cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    def calibrate_camera(self):
        for fname in os.listdir(self.out_folder):
            if fname.endswith(".jpg"):
                img_path = os.path.join(self.out_folder, fname)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Irgendein Fehler beim Laden des Bildes")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = self.find_corners(gray)
                if ret:
                    obj_point = self.create_object_points()
                    self.obj_points.append(obj_point)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
                    self.img_points.append(corners2)
                    self.img_names.append(fname)
                    cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
                    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                    cv2.imshow('img', img)
                    cv2.waitKey(50)

        cv2.destroyAllWindows()

        # Kamerakalibrierung (mit den Objekt- und Bildpunkten)
        self.ret, self.cam_matrix, self.distortion_coeff, self.rotation_vectors, self.translation_vectors = cv2.calibrateCamera(
            self.obj_points, self.img_points, gray.shape[::-1], None, None)

        # Kalibrierungsergebnisse speichern
        np.savez(self.calib_file,
                 cam_matrix=self.cam_matrix,
                 distortion_coeff=self.distortion_coeff,
                 rotation_vectors=self.rotation_vectors,
                 translation_vectors=self.translation_vectors)

    def calculate_reprojection_error(self):
        mean_error = 0
        for i in range(len(self.obj_points)):
            imgpoints2, _ = cv2.projectPoints(self.obj_points[i], self.rotation_vectors[i], self.translation_vectors[i], self.cam_matrix, self.distortion_coeff)
            error = cv2.norm(self.img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        mean_error /= len(self.obj_points)
        print(f"Gesamter Reprojektionsfehler: {mean_error}")

    def load_calibration_data(self):
        calib_data = np.load(self.calib_file)
        self.cam_matrix = calib_data['cam_matrix']
        self.distortion_coeff = calib_data['distortion_coeff']
        self.rotation_vectors = calib_data['rotation_vectors']
        self.translation_vectors = calib_data['translation_vectors']

    def print_calibration_results(self):
        print("Kalibrierung wurde abgeschlossen!")
        print("Erfolgsrate (RMS-Fehler):", self.ret)
        print("Kameramatrix:")
        print(self.cam_matrix)
        print("Verzerrungskoeffizienten:")
        print(self.distortion_coeff)
        print("Rotationsvektoren:")
        print(self.rotation_vectors)
        print("Translationsvektoren:")
        print(self.translation_vectors)


class ImageUndistorter:
    """
    Ist für das Entzerren von Bildern verantwortlich und entfernt Verzerrungen basierend auf der Kameramatrix und den Verzerrungskoeffizienten
    (die während der Kamera-Kalibrierung berechnet wurden) und speichert das entzerrte Bild.
    """
    def __init__(self, in_folder, out_folder, cam_matrix, distortion_coeff):
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.cam_matrix = cam_matrix
        self.distortion_coeff = distortion_coeff

    def undistort_image(self, image_name, output_name):
        img_path = os.path.join(self.in_folder, image_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Fehler: Bild '{image_name}' konnte nicht geladen werden.")
            return None

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.cam_matrix, self.distortion_coeff, (w, h), 1, (w, h))

        # Verzerrung rausnehmen
        dst = cv2.undistort(img, self.cam_matrix, self.distortion_coeff, None, newcameramtx)

        # Bild zuschneiden
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        # Bild speichern
        undistorted_img_path = os.path.join(self.out_folder, output_name)
        cv2.imwrite(undistorted_img_path, dst)

        return undistorted_img_path


class PointProjector:
    """
    Ist für das Laden von Punktdaten der RK Logdatei verantwortlich (2D-Pixelkoordinaten (Sample-Punkte)), die Zielkoordinaten (3D-Target-Punkte) und 3D-berechnete Punkte aus dem RK Algorithmus.
    und in anpasst (XYZ-Konvention einhält) um für die weitere Berechnung nutzbar gemacht werden zu können.
    """
    def __init__(self, cam_matrix, distortion_coeff):
        self.cam_matrix = cam_matrix
        self.distortion_coeff = distortion_coeff

    def project(self, points):
        rvec = tvec = np.array([0.0, 0.0, 0.0])
        projected_points, _ = cv2.projectPoints(points, rvec, tvec, self.cam_matrix, self.distortion_coeff)
        return np.squeeze(projected_points, axis=1)

    def unproject(self, points, Z):
        points_undistorted = cv2.undistortPoints(np.expand_dims(points.astype(np.float32), axis=1), self.cam_matrix, self.distortion_coeff, P=self.cam_matrix)
        points_undistorted = np.squeeze(points_undistorted, axis=1)

        f_x = self.cam_matrix[0, 0]
        f_y = self.cam_matrix[1, 1]
        c_x = self.cam_matrix[0, 2]
        c_y = self.cam_matrix[1, 2]

        result = []
        for idx in range(points_undistorted.shape[0]):
            z = Z[0] if len(Z) == 1 else Z[idx]
            x = (points_undistorted[idx, 0] - c_x) / f_x * z
            y = (points_undistorted[idx, 1] - c_y) / f_y * z
            result.append([x, y, z])

        return np.array(result)

    def get_single_projection(self, u, v, z=Z):
        point = np.array([[u, v]], dtype=np.float32)
        Z = np.array([z], dtype=np.float32)
        return self.unproject(point, Z)[0]

    def get_multiple_projections(self, points):
        results = []
        for (u, v) in points:
            results.append(self.get_single_projection(u, v))
        return results


class PointLoader:
    @staticmethod
    def get_sample_points(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        sample_points = []

        for line in lines:
            groups = line.split('][')
            if len(groups) < 3:
                continue
            sample = groups[0].replace('[', '').replace(']', '').strip()
            sample_points.append(sample)

        if sample_points[0].lower() == 'sample':
            sample_points.pop(0)

        uv_points = []
        for point in sample_points:
            u, v = point.split(',')
            uv_points.append((int(u.strip()), int(v.strip())))

        return uv_points

    @staticmethod
    def get_target_points(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        target_points = []

        for line in lines:
            groups = line.split('][')
            if len(groups) < 3:
                continue
            target = groups[1].replace('[', '').replace(']', '').strip()
            target_points.append(target)

        if target_points[0].lower() == 'target':
            target_points.pop(0)

        uv_points = []
        for point in target_points:
            u, v, z = map(float, point.split(','))
            # Vertausche v und z, runde auf 2 Dezimalstellen
            uv_points.append((round(u, 2), round(-z, 2), round(v, 2)))

        return uv_points

    @staticmethod
    def get_computed_points(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        computed_points = []

        for line in lines:
            groups = line.split('][')
            if len(groups) < 3:
                continue
            computed = groups[2].replace('[', '').replace(']', '').strip()
            computed_points.append(computed)

        if computed_points[0].lower() == 'computed':
            computed_points.pop(0)

        uv_points = []
        for point in computed_points:
            u, v, z = map(float, point.split(','))
            # Vertausche v und z, runde auf 2 Dezimalstellen
            uv_points.append((round(u, 2), round(-z, 2), round(v, 2))) 

        return uv_points



class ImageMarker:
    """
    Ist für das Zeichnen von den projizierten Punkten auf Bildern verantwortlich und nimmt die aus den 2D Pixelpunkte (Sample) und berechnete Punkte und udn den Zielkoordinaten 
    und zeichnet diese auf das Bild.
    """
    def __init__(self, in_folder, out_folder):
        self.in_folder = in_folder
        self.out_folder = out_folder

    def draw_points(self, image_name, points_sample, points_target, points_computed, output_name):
        img_path = os.path.join(self.in_folder, image_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Fehler: Bild '{image_name}' konnte nicht geladen werden.")
            return

        # Zeichne die projizierten Sample-Punkte auf das Bild (grün)
        for point in points_sample:
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)  # Green color in BGR

        # Zeichne die projizierten Target-Punkte auf das Bild (blau)
        for point in points_target:
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)  # Blue color in BGR
            #print(point)

        # Zeichne die projizierten Computed-Punkte auf das Bild (rot)
        for point in points_computed:
            cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)  # Red color in BGR

        # Legende
        legend_x, legend_y = 20, 40
        cv2.putText(img, 'Sample Points (mit OpenCV-Algorithmus berechnet)', (legend_x, legend_y), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
        cv2.putText(img, 'Target Punkte (aus RK Datei)', (legend_x, legend_y + 45), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        cv2.putText(img, 'Computed Punkte (aus RK Datei)', (legend_x, legend_y + 90), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        # Ergebnis speichern und anzeigen
        output_image_path = os.path.join(self.out_folder, output_name)
        cv2.imwrite(output_image_path, img)

        cv2.namedWindow('marked_img', cv2.WINDOW_NORMAL)
        cv2.imshow('marked_img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class ErrorCalculator:
    """
    Berechnung von Metriken (Fehlermaßen) zwischen berechneten und den Referenzwerten (Zielkoordinaten). 
    MAE, RMSE, Euklidische Distanz und Differenzen (Deltas) zwischen den Punkten berechnen und teils visuell plotten.
    """
    @staticmethod
    def calculate_deltas(computed_points, target_points):
        return np.array([np.subtract(target, computed) for target, computed in zip(target_points, computed_points)])

    # MAE: Durchschnittliche absolute Differenz (MAE) zwischen den berechneten Punkten und den Zielpunkten
    @staticmethod
    def calculate_mae(deltas):
        return np.mean(np.abs(deltas), axis=0)

    @staticmethod
    def calculate_total_mae(deltas):
        return np.mean(ErrorCalculator.calculate_mae(deltas))

    # RMSE: Differenzen quadrieren, den Mittelwert berechnen und anschließend die Quadratwurzel ziehen
    @staticmethod
    def calculate_rmse(deltas):
        return np.sqrt(np.mean(np.square(deltas), axis=0))

    @staticmethod
    def calculate_total_rmse(deltas):
        return np.mean(ErrorCalculator.calculate_rmse(deltas))

    # Euklidische Distanz: Euklidische-Norm zwischen den berechneten und den Zielpunkten verwenden
    @staticmethod
    def calculate_euclidean_distance(deltas):
        return np.linalg.norm(deltas, axis=1)

    @staticmethod
    def calculate_total_euclidean_distance(deltas):
        return np.mean(ErrorCalculator.calculate_euclidean_distance(deltas))

    # Plotly Funktionen zwecks Plottings
    @staticmethod
    # Visualisierung der Delta Werte in X-Richtung
    def plot_x_deltas(deltas, title, output_file):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=deltas[:, 0], mode='lines+markers', name='Delta X', line=dict(color='red')))
        fig.add_shape(type="line", x0=0, y0=0, x1=len(deltas), y1=0, line=dict(color="black", width=2, dash="dash"))
        max_delta = np.max(np.abs(deltas[:, 0]))
        fig.update_layout(
            title=title,
            xaxis_title="Punkte",
            yaxis_title="X-Abweichung",
            legend_title="Achse",
            template="plotly_white",
            yaxis=dict(range=[-max_delta - 1, max_delta + 1])
        )
        fig.write_image(output_file)
        fig.show()

    @staticmethod
    def plot_y_deltas(deltas, title, output_file):
        # Visualisierung der Delta Werte in Y-Richtung
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=deltas[:, 1], mode='lines+markers', name='Delta Y', line=dict(color='blue')))
        fig.add_shape(type="line", x0=0, y0=0, x1=len(deltas), y1=0, line=dict(color="black", width=2, dash="dash"))
        max_delta = np.max(np.abs(deltas[:, 1]))
        fig.update_layout(
            title=title,
            xaxis_title="Punkte",
            yaxis_title="Y-Abweichung",
            legend_title="Achse",
            template="plotly_white",
            yaxis=dict(range=[-max_delta - 1, max_delta + 1])
        )
        fig.write_image(output_file)
        fig.show()

    def compare_algorithms(self, computed_points_opencv, computed_points_reknow, target_points):
        # Berechnung der Deltas
        deltas_opencv = self.calculate_deltas(computed_points_opencv, target_points)
        deltas_reknow = self.calculate_deltas(computed_points_reknow, target_points)

        # Berechnung der Fehlermaße
        mae_opencv = round(self.calculate_total_mae(deltas_opencv), 2)
        rmse_opencv = round(self.calculate_total_rmse(deltas_opencv), 2)
        euclidean_opencv = round(self.calculate_total_euclidean_distance(deltas_opencv), 2)
        
        mae_reknow = round(self.calculate_total_mae(deltas_reknow), 2)
        rmse_reknow = round(self.calculate_total_rmse(deltas_reknow), 2)
        euclidean_reknow = round(self.calculate_total_euclidean_distance(deltas_reknow), 2)

        # Ergebnisse
        print(f"OpenCV - MAE: {mae_opencv}, RMSE: {rmse_opencv}, Euklidische Distanz: {euclidean_opencv}")
        print(f"REKNOW - MAE: {mae_reknow}, RMSE: {rmse_reknow}, Euklidische Distanz: {euclidean_reknow}")

        return mae_opencv, rmse_opencv, euclidean_opencv, mae_reknow, rmse_reknow, euclidean_reknow




def main():
    # Output-Folder erstellen (falls nicht vorhanden)
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    if not os.path.exists(CALIB_FILE):
        # Bilder bearbeiten und speichern
        processor = ImageProcessor(IN_FOLDER, OUT_FOLDER)
        processor.process_images()

        # Kamera kalibrieren
        calibrator = CameraCalibration(OUT_FOLDER, CALIB_FILE)
        calibrator.calibrate_camera()
        calibrator.calculate_reprojection_error()

        # Kalibrierungsergebnisse ausgeben
        # calibrator.print_calibration_results()
    else:
        # Kalibrierungsergebnisse laden
        calibrator = CameraCalibration(OUT_FOLDER, CALIB_FILE)
        calibrator.load_calibration_data()

    # Bild entzerren und speichern
    undistorter = ImageUndistorter(IN_FOLDER, OUT_FOLDER, calibrator.cam_matrix, calibrator.distortion_coeff)
    undistorted_img_path = undistorter.undistort_image("240500013_markings.png", "undistorted_img.png")

    if undistorted_img_path is None:
        print("Fehler beim Entzerren des Bildes.")
        return

    # Punktprojektionen und -umprojektionen
    point_projector = PointProjector(calibrator.cam_matrix, calibrator.distortion_coeff)

    # RK-Logdatei mit Punkten einlesen
    file_path = 'a_lot_of_points.txt'

    # Eingelese RK Logdatei (Sample (2D), Target (3D), Computed (3D)) 
    read_sample_points = PointLoader.get_sample_points(file_path)
    target_points = PointLoader.get_target_points(file_path)
    computed_points = PointLoader.get_computed_points(file_path)

    # Ausgabe der Punkte
    print("RK Sample Punkte:", read_sample_points[:10])
    print("RK Target Punkte:", target_points[:10])
    print("RK Computed Punkte:", computed_points[:10])
    print("****")

    # Berechnung der 3D-Koordinaten (hier sample u,v Koordinaten verwenden mit statisch z = Z) aus den 2D-Punkten aus RK
    calculated_opencv_3d = point_projector.get_multiple_projections(read_sample_points)

    # Runden der berechneten 3D-Koordinaten auf zwei Dezimalstellen
    calculated_opencv_3d = np.round(calculated_opencv_3d, 2)

    # Ausgabe der berechneten 3D-Koordinaten mit den Sample Points (u, v)
    for i, (u, v) in enumerate(read_sample_points):
        calculated_projection = calculated_opencv_3d[i]
        print(f"Point {i} ({u}, {v}): {calculated_projection}")

    # 3D-Punkte in Numpy-Array umrechnen + runden auf zwei Dezimalstellen
    points_3d_opencv = np.round(np.array(calculated_opencv_3d), 2)
    points_3d_target = np.round(np.array(target_points), 2)
    points_3d_computed = np.round(np.array(computed_points), 2)

    # Ausgabe der ersten 10 Punkte
    print("\nDie ersten 10 Punkte in points_3d_target (RK Zielkoordinaten):")
    for i in range(10):
        print(f"Point {i} (Target RK): ({points_3d_target[i][0]}, {points_3d_target[i][1]}, {points_3d_target[i][2]})")

    print("\nDie ersten 10 Punkte in points_3d_computed (RK berechnete 3D-Koordinaten):")
    for i in range(10):
        print(f"Point {i} (Computed RK): ({points_3d_computed[i][0]}, {points_3d_computed[i][1]}, {points_3d_computed[i][2]})")

    print("\nDie ersten 10 Punkte in points_3d_opencv (OpenCV berechnete 3D-Koordinaten):")
    for i in range(10):
        print(f"Point {i} (Computed OpenCV): ({points_3d_opencv[i][0]}, {points_3d_opencv[i][1]}, {points_3d_opencv[i][2]})")

    # 3D-Punkte auf 2D-Punkte projizieren
    projected_3d_opencv_calculated = point_projector.project(points_3d_opencv)
    projected_3d_target_read = point_projector.project(points_3d_target)
    projected_3d_computed_read = point_projector.project(points_3d_computed)

    # Punkte auf dem Testbild markieren und speichern
    image_marker = ImageMarker(IN_FOLDER, OUT_FOLDER)
    image_marker.draw_points("240500013_markings.png", projected_3d_opencv_calculated, projected_3d_target_read, projected_3d_computed_read, "drawn_points_img.png")


    # **** **** **** **** **** 


    # Fehlerberechnungen
    error_calculator = ErrorCalculator()
    mae_opencv, rmse_opencv, euclidean_opencv, mae_reknow, rmse_reknow, euclidean_reknow = error_calculator.compare_algorithms(points_3d_opencv, points_3d_computed, points_3d_target)

    # Berechnung der Deltas
    deltas_opencv = ErrorCalculator.calculate_deltas(points_3d_opencv, points_3d_target)
    deltas_java = ErrorCalculator.calculate_deltas(points_3d_computed, points_3d_target)

    # Plotten der X-Deltas
    ErrorCalculator.plot_x_deltas(deltas_opencv, 'X-Deltas zwischen OpenCV berechneten Punkten und Zielpunkten', 'x_deltas_opencv.png')
    ErrorCalculator.plot_x_deltas(deltas_java, 'X-Deltas zwischen Java berechneten Punkten und Zielpunkten', 'x_deltas_java.png')

    # Plotten der Y-Deltas
    ErrorCalculator.plot_y_deltas(deltas_opencv, 'Y-Deltas zwischen OpenCV berechneten Punkten und Zielpunkten', 'y_deltas_opencv.png')
    ErrorCalculator.plot_y_deltas(deltas_java, 'Y-Deltas zwischen Java berechneten Punkten und Zielpunkten', 'y_deltas_java.png')



    """
    for i in range(len(target_points[:10])):
        delta_x = target_points[i][0] - computed_points[i][0]
        delta_y = target_points[i][1] - computed_points[i][1]
        delta_z = target_points[i][2] - computed_points[i][2]

        delta_x_opencv = target_points[i][0] - calculated_opencv_3d[i][0]
        delta_y_opencv = target_points[i][1] - calculated_opencv_3d[i][1]
        delta_z_opencv = target_points[i][2] - calculated_opencv_3d[i][2]

        print(f"Target Point {i} (RK): {target_points[i]}")
        print(f"Computed Point {i} (RK): {computed_points[i]}")
        print(f"Computed Point {i} (OpenCV): {calculated_opencv_3d[i]}")
        print(f"Delta (RK): (Delta X: {delta_x}, Delta Y: {delta_y}, Delta Z: {delta_z})")
        print(f"Delta (OpenCV): (Delta X: {delta_x_opencv}, Delta Y: {delta_y_opencv}, Delta Z: {delta_z_opencv})")
        print(f"***")


        print(f"Target Point {i} (RK): {target_points[i]}")
        print(f"Computed Point {i} (RK): {computed_points[i]}")
        print(f"Delta (RK): (Delta X: {delta_x}, Delta Y: {delta_y}, Delta Z: {delta_z})")
        print(f"Delta (OpenCV): (Delta X: {delta_x}, Delta Y: {delta_y}, Delta Z: {delta_z})")
        print(f"***")
    """

    """
    # Punkte der Realmessung
    points = [(903, 1067), (973, 1069), (1042, 1070), (1111, 1070), (1180, 1072)]

    # Berechnung der 3D-Punkte
    calculated_projections = point_projector.get_multiple_projections(points)

    # Ausgabe berechneten 3D-Koordinaten
    for i, point in enumerate(calculated_projections):
        print(f"3D-Koordinate für Punkt {i+1}: {point}")
    """

if __name__ == "__main__":
    main()