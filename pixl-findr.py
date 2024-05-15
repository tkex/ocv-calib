import cv2
import os

# Aktuelle Arbeitsverzeichnis setzen
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Pfad (relatov)
img_dir = 'imgs'
img_name = 'LUCID_PHX050S-C_240400411__20240503161959760_image0.jpg'

img_path = os.path.join(img_dir, img_name)

# Bild laden
image = cv2.imread(img_path)

# Koordinaten (x,y)
x, y = 50, 100

if image is None:
    print(f"Das Bild konnte nicht geladen werden: {img_path}")
else:

    # Sind Koordinaten innerhalb des Bildes
    # TODO: Zusätzlichen Check für außerhalb der Koordinaten-Grenzen ergänzen
    if x < 0 or y < 0:
        print(f"Koordinaten ({x}, {y}) liegen unterhalb des Bildes.")
    else:
        # Pixelwert an diesen Koordinaten
        pixel_value = image[y, x]

        print(f"Pixelwert an Koordinate ({x}, {y}): {pixel_value}")

        # Zeige Bild und Koordinaten farblich markieren (grün)
        img_marked = image.copy()
        cv2.circle(img_marked, (x, y), 5, (0, 255, 0), -1)

        # Speichern des Bilds
        out_dir = 'pixl-output'
        os.makedirs(out_dir, exist_ok=True)
        out_img_path = os.path.join(out_dir, 'img_markiert.jpg')

        cv2.imwrite(out_img_path, img_marked)

        print(f"Bild gespeichert: {out_img_path}")

        # Zeige Bild
        cv2.imshow('Pixelkoordinaten', img_marked)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
