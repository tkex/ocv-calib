import cv2
import os

# Aktuelles Verzeichnis
project_dir = os.path.dirname(os.path.abspath(__file__))

# Pfade der Bilder
input_folder = os.path.join(project_dir, 'imgs')
output_folder = os.path.join(project_dir, 'imgs_edited')

# Output-Folder erstellen (falls nicht vorhanden)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Bearbeiten der Bilder
def edit_image(img):

    # Bild in Graustufen umwandeln
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kontrast erhöhen
    edited = cv2.equalizeHist(gray)

    return edited

# Bilder bearbeiten und speichern
for filename in os.listdir(input_folder):

    if filename.endswith(".jpg"):

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        edited_img = edit_image(img)
        edited_img_path = os.path.join(output_folder, filename)

        cv2.imwrite(edited_img_path, edited_img)
