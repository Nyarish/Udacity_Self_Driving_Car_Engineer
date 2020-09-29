# Get imports
import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Helper functions
import detect_lane_pixels
import perspective_transform



def get_lane_curvature_real(image):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    binary_warped = np.copy(image)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    _, left_fitx, right_fitx, ploty = detect_lane_pixels.search_around_poly(binary_warped)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
       
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    image_width = binary_warped.shape[1]
    
    image_centerx = image_width // 2
    lane_width = abs(left_fitx[-1] - right_fitx[-1])
    lane_center_x = (left_fitx[-1] + right_fitx[-1]) // 2
    pix_offset = image_centerx - lane_center_x
    
    lane_width_m = 3.7  # How wide we expect the lane to be in meters
    
    # Lane mean position relative to image center. 
    # If positive, the lane is offset to the left i.e. positive vehicle is to the right.
    offset = lane_width_m * (pix_offset/lane_width)
        
    return left_curverad, right_curverad, offset




def add_text(original_img, top_down):
    
    top_down = np.copy(top_down)
    img = np.copy(original_img)
    
    left_curverad, right_curverad, offset = get_lane_curvature_real(top_down)
    
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = .7
    color =  (255, 255, 255)
    line_type = cv2.LINE_AA
    cv2.putText(
        img,
        'Left Curvature: {:.2f}m, Right Curvature: {:.2f}m'.format(left_curverad, right_curverad),
        (20, 40),  # origin point
        font,
        scale,
        color,
        lineType=line_type
    )
    cv2.putText(
        img,
        'Center-Lane Offset to Vehicle position: {:.2f}m'.format(offset),
        (20, 80),  # origin point
        font,
        scale,
        color,
        lineType=line_type
    )
    
    
    return img



def draw_lane_to_orinalImage(undist_image, binary_warped):
    
    top_down = np.copy(binary_warped)
    image = np.copy(undist_image)
    
    _, left_fitx, right_fitx, ploty = detect_lane_pixels.search_around_poly(top_down)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(top_down).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp  = perspective_transform.get_original_perspective(color_warp)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # Combine the result with the original image
    original_img = cv2.addWeighted(image, 1, newwarp, 0.3, 0)# 0.3, 0
    
    marked_lane = add_text(original_img, top_down)
    
    return marked_lane


def get_color_zone_warp(top_down):
    
    undist_img = np.copy(top_down)
    
    _, left_fitx, right_fitx, ploty = detect_lane_pixels.search_around_poly(top_down)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(undist_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    return color_warp

