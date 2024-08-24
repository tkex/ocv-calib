# Camera Calibration and 3D Point Projection

This project provides tools for camera calibration, image processing, and 3D point projection using OpenCV. It includes functions for calibrating a camera using chessboard images, undistorting images, projecting 3D points onto a 2D image plane and vice versa, and comparing results via evaluation metrics (RMSE, MAE, Euclidean distances, delta differences). The intention of the project is to compare the calibration between OpenCV and a custom commercial Java implementation without any framework whatsoever. To achieve this, a log file is read in containing 2D (uv) points, computed points from the commercial Java implementation, and real target points (measured and accurate). The results from the OpenCV calibration are then compared to the data from the Java implementation.


## Project Structure

- **imgs/**: Folder for your input images.
- **imgs_edited/**: Folder where processed and undistorted images will be saved.
- **calibration_data.npz**: File where camera calibration data is stored. Needs to be deleted if you calibrate it with new images.
- ***a_lot_of_points.txt**: Log file from the commerical Java project to compare the results from the OpenCV calibration against.

## Key Features

- **Image Processing:** Rotate and convert images to grayscale.
- **Camera Calibration:** Calibrate camera using chessboard images and saving the calibration results for future use in own calibration file.
- **Image Undistortion:** Remove distortions from images based on the calibration data.
- **3D Point Projection:** Project 3D points to 2D coordinates and back (using the cameras intrinsic parameters).
- **Error Analysis:** Compare and visualize the accuracy of computed 3D points against reference targets.

## How to Use

1. **Prepare your images:**
   - Place your calibration images in the `imgs` folder. The project comes with some default images.
   - The script will process these images, convert them to grayscale, and save them in the `imgs_edited` folder.

2. **Run Camera Calibration:**
   - The script calibrates the camera using the processed images, generating calibration data that will be stored for later use.

3. **Undistort Images:**
   - The script can undistort images using the generated calibration data, saving the results back to the `imgs_edited` folder.

4. **3D Point Projection:**
   - Use the tools to project 3D points onto a 2D plane or reproject 2D points back into 3D space.

5. **Compare Algorithms:**
   - The script compares the two different algorithms by calculating error metrics and visualizing the differences between the OpenCV computed points 3D (from 2D) and the computed 3D Java points and its reference target points.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- Plotly
    
```bash
pip install opencv-python numpy plotly


## Running the Project

To run the project, simply execute the `main()` function.

```bash
python ocv-calib.py
