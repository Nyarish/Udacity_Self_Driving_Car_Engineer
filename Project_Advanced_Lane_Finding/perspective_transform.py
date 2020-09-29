# Get imports
import numpy as np
import cv2
import glob
import math 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# # Pick four points in a trapezoidal on straight lines
# def get_trapezoid(image):
    
#     img = np.copy(image)
#     img_size = (img.shape[1], img.shape[0])
    
#     src = np.float32(
#     [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
#     [((img_size[0] / 6) - 10), img_size[1]],
#     [(img_size[0] * 5 / 6) + 60, img_size[1]],
#     [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    
#     dst = np.float32(
#     [[(img_size[0] / 4), 0],
#     [(img_size[0] / 4), img_size[1]],
#     [(img_size[0] * 3 / 4), img_size[1]],
#     [(img_size[0] * 3 / 4), 0]])
    
    
#     return src, dst




def get_trapezoid(img): #warp_to_lines(img)
    ''' "img" should be an undistorted image. '''
    
    x_shape, y_shape = img.shape[1], img.shape[0]
    middle_x = x_shape//2
    top_y = 2*y_shape//3
    top_margin = 93
    bottom_margin = 450
    points = [
        (middle_x-top_margin, top_y),
        (middle_x+top_margin, top_y),
        (middle_x+bottom_margin, y_shape),
        (middle_x-bottom_margin, y_shape)
    ]


    src = np.float32(points)
    dst = np.float32([
        (middle_x-bottom_margin, 0),
        (middle_x+bottom_margin, 0),
        (middle_x+bottom_margin, y_shape),
        (middle_x-bottom_margin, y_shape)
    ])
        
    return src,dst










def get_original_perspective(image):
    img = np.copy(image)
    y, x = img.shape[0], img.shape[1]  # Image size
    src, dst = get_trapezoid(img)

    # Get inverse perspective transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Reverse the top-down perspective transformation with inverse matrix
    unwarped = cv2.warpPerspective(img, Minv, (x, y), flags=cv2.INTER_LINEAR)

    return unwarped


def get_transformed_perspective(image):
    img = np.copy(image)
    y, x = img.shape[0], img.shape[1] # Image size
    src, dst = get_trapezoid(img)

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Calculate top-down image using perspective transform matrix
    warped = cv2.warpPerspective(img, M, (x, y), flags=cv2.INTER_LINEAR)

    return warped

def visualizePerspectiveTransform(undist_img, warped_img):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    ax1.imshow(undist_img)
    ax1.set_title('Undistorted Image', fontsize=15)
    ax2.imshow(warped_img)
    ax2.set_title('Undistorted and Top_down Image', fontsize=15)

