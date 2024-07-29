import cv2
import numpy as np
import os


# Konstanten
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_FOLDER = os.path.join(PROJECT_DIR, 'imgs')
OUT_FOLDER = os.path.join(PROJECT_DIR, 'imgs_edited')
CALIB_FILE = os.path.join(OUT_FOLDER, 'calib_data.npz')
CHESSBOARD_SIZE = (8, 4)
SQUARE_SIZE = 28.57
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
Z = 171  # Tiefe (Z) für die Umprojektion (2D PK <-> 3D K)

class ImageProcessor:
    def __init__(self, in_folder, out_folder):
        self.in_folder = in_folder
        self.out_folder = out_folder

    def better_img_edit(self, img):
        # Bild drehen gegen den Uhrzeigersinn (90 Grad)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Bild in Graustufen umwandeln
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def process_images(self):
        for fname in os.listdir(self.in_folder):
            if fname.endswith(".jpg"):
                img_path = os.path.join(self.in_folder, fname)
                img = cv2.imread(img_path)
                edited_img = self.better_img_edit(img)
                edited_img_path = os.path.join(self.out_folder, fname)
                cv2.imwrite(edited_img_path, edited_img)


class CameraCalibration:
    def __init__(self, out_folder, calib_file):
        self.out_folder = out_folder
        self.calib_file = calib_file
        # 3D Punkte in der realen Welt
        self.obj_points = []
        # 2D Punkte in den Bildern
        self.img_points = []
        # Liste um Namen zu speichern für RMS (Bild <-> RMS Zuweisung)
        self.img_names = []
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
            # Vertausche v und z
            uv_points.append((int(u), -int(z), int(v)))

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
            # Vertausche v und z
            uv_points.append((int(u), -int(z), int(v)))

        return uv_points


class ImageMarker:
    def __init__(self, in_folder, out_folder):
        self.in_folder = in_folder
        self.out_folder = out_folder

    def draw_points(self, image_name, points_sample, points_target, points_computed, output_name):
        img_path = os.path.join(self.in_folder, image_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Fehler: Bild '{image_name}' konnte nicht geladen werden.")
            return

        # Zeichne der projizierten Sample-Punkte auf das Bild (grün)
        for point in points_sample:
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)  # Green color in BGR

        # Zeichne der projizierten Target-Punkte auf das Bild (blau)
        for point in points_target:
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)  # Blue color in BGR
            #print(point)

        # Zeichne der projizierten Computed-Punkte auf das Bild (rot)
        for point in points_computed:
            cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)  # Red color in BGR

        # Legende hinzufügen
        legend_x, legend_y = 20, 40
        cv2.putText(img, 'Sample Points (mit OpenCV-Algorithmus berechnet)', (legend_x, legend_y), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
        cv2.putText(img, 'Target Punkte (aus RK Datei)', (legend_x, legend_y + 45), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        cv2.putText(img, 'Computed Punkte (aus RK Datei)', (legend_x, legend_y + 90), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        # Markiertes Bild speichern und anzeigen
        output_image_path = os.path.join(self.out_folder, output_name)
        cv2.imwrite(output_image_path, img)

        cv2.namedWindow('marked_img', cv2.WINDOW_NORMAL)
        cv2.imshow('marked_img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # Output-Folder erstellen (sofern nicht vorhanden)
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

        # Kalibrierungsergebnisse ausgeben (auskommentiert)
        # calibrator.print_calibration_results()
    else:
        # Kalibrierungsergebnisse laden
        calibrator = CameraCalibration(OUT_FOLDER, CALIB_FILE)
        calibrator.load_calibration_data()

    # Bild entzerren und speichern
    undistorter = ImageUndistorter(IN_FOLDER, OUT_FOLDER, calibrator.cam_matrix, calibrator.distortion_coeff)
    undistorted_img_path = undistorter.undistort_image("240500013_markings.png", "entzerrtes_bild.png")

    if undistorted_img_path is None:
        print("Fehler beim Entzerren des Bildes.")
        return

    # Punktprojektionen und -umprojektionen
    point_projector = PointProjector(calibrator.cam_matrix, calibrator.distortion_coeff)

    # Datei mit den Punkten einlesen
    file_path = 'a_lot_of_points.txt'
    sample_points = PointLoader.get_sample_points(file_path)
    target_points = PointLoader.get_target_points(file_path)
    computed_points = PointLoader.get_computed_points(file_path)

    # Ausgabe der Punkte (auskommentiert)
    # print("Sample Points:", sample_points)
    # print("Target Points:", target_points)
    # print("Computed Points:", computed_points)

    # Ausgabe der Target Points (auskommentiert)
    # for i, point in enumerate(target_points):
    #    print(f"Target Point {i}: {point}")

    # Berechnung der Projektionen (hier sample u,v Koordinaten verwenden mit statisch z = Z (171))
    calculated_projections = point_projector.get_multiple_projections(sample_points)

    # Ausgabe der berechneten 3D-Koordinaten (auskommentiert)
    # for i, point in enumerate(calculated_projections):
    #     print(f"Sample Point {i}: {point}")

    # 3D-Punkte in NumPy-Arrays umwandeln
    points_3d_sample = np.array(calculated_projections, dtype=np.float32)
    points_3d_target = np.array(target_points, dtype=np.float32)
    points_3d_computed = np.array(computed_points, dtype=np.float32)

    # 3D-Punkte auf 2D-Punkte projizieren
    projected_points_sample = point_projector.project(points_3d_sample)
    projected_points_target = point_projector.project(points_3d_target)
    projected_points_computed = point_projector.project(points_3d_computed)

    # Punkte auf dem Originalbild markieren + speichern
    image_marker = ImageMarker(IN_FOLDER, OUT_FOLDER)
    image_marker.draw_points("240500013_markings.png", projected_points_sample, projected_points_target, projected_points_computed, "eingezeichnete_punkte.png")


if __name__ == "__main__":
    main()
