# imports 

import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Helper functions for camera calibration

# Directory for calibration images
image_dir = glob.glob('camera_cal/calibration*.jpg')
# Directory to Save objpoints and imgpoints on pickle
camera_cal_fname = './camera_cal/output_images/camera_cal.p'

def calibration():
    
    """ 
    This function takes in a set of images used for calibration,
    and outputs objpoints, imgpoints and corners to compute the 
    camera calibration and distortion coefficients using the cv2.calibrateCamera() function.

    input: images
    args: 
    output:objpoints, imgpoints, corners
    
    """
    # To be used later to count number of images calibrated
    images_calibrated = []
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = image_dir
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            images_calibrated.append(img)

            # Draw and save the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            write_name = 'Corners_found'+str(idx)+'.jpg'
            cv2.imwrite('./camera_cal/output_images/'+ write_name, img)
            
            
    cv2.destroyAllWindows()
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # save to pickle
    camera_cal_dict = {'mtx':mtx, 'dist':dist}
    pickle.dump(camera_cal_dict, open(camera_cal_fname, "wb"))
    
    print('Camera calibrated using {0} images'.format(np.array(images_calibrated).shape[0]))


    
def get_calibration_matrix():
    
    try:
        dist_pickle = pickle.load( open( camera_cal_fname, "rb" ) )
        
    except FileNotFoundError:
        
        calibration()
        dist_pickle = pickle.load( open( camera_cal_fname, "rb" ) )
    
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    return mtx, dist
        