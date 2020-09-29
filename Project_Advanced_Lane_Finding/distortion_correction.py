# imports 
import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Helper functions 

def undistort_image(image, mtx, dist):
    
    """
    This function accepts an image, camera matrix as mtx, and distortion coefficients as dist.
    The function uses (cv2.undistort(image, mtx, dist, None, mtx)) to undistort then image,and
    returns a undistorted image.
    
    inputs: image, mtx, dist
    args:(cv2.undistort(image, mtx, dist, None, mtx))
    returns: undistorted image
    
    """
    
    return cv2.undistort(image, mtx, dist, None, mtx)


def get_undistorted_image(image, mtx, dist, gray_scale=False):
    
    img_undistorted = undistort_image(image, mtx, dist)
    
    if gray_scale:
        return cv2.cvtColor(img_undistorted, cv2.COLOR_RGB2GRAY)
    else:
        return img_undistorted


def read_undistorted_image_fname(fname, mtx, dist):
    
    """
    This function takes image filename as fname, camera matrix as mtx, and distortion coefficients as dist.
    The function undistorts the image using (cv2.undistort(image, mtx, dist, None, mtx)), 
    writes and saves the undistorted image to a directory.
    
    
    inputs: image, mtx, dist
    args:(cv2.undistort(image, mtx, dist, None, mtx)),
         write undist using (cv2.imwrite())
   
   returns: undistorted image
    
    """
    
    # Read image
    img = mpimg.imread(fname)
    image = np.copy(img)
    
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    undist_img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    
    
    # Create output folder if missing 
    image_dir = './output_images/'
    
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    
    # write image 
    name = fname.split('/')[-1]
    save_name = 'Undist_'+name
    cv2.imwrite(image_dir+save_name,undist_img)
    
    
    # Read saved image
    fname = image_dir+save_name
    undist_img = mpimg.imread(fname)
    
    return undist_img

    
def visualizeUndistortion(fname, undist_img):
    #original_image = mpimg.imread(fname)
    
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(fname);
    ax1.set_title('Original Image', fontsize=30)

    ax2.imshow(undist_img);
    ax2.set_title('Undistorted Image', fontsize=30)