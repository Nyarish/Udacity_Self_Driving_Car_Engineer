

**Advanced Lane Finding Project**

---
In this project, the objective is to detect lane lines in images using Python and OpenCV. OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.

The goal and steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
---
[//]: # (Image References)

[image1]: ./output_images/calibrated_undistorted_image.jpeg "calibrated_undistorted_image"  
[image2]: ./output_images/original_vs_undistorted.jpeg "Road Transformed"
[image3]: ./output_images/color_binary_image.jpeg "Binary Example"
[image4]: ./output_images/pespective_transform_src_and_dst.jpeg "Warp Example"
[image5]: ./output_images/sliding_window_lanes.jpeg "sliding_window_lanes"
[image6]: ./output_images/search_around_poly.jpeg "search_around_poly"  
[image7]: ./output_images/output.jpeg "Output" 

[video1]: ./videos_out/project_video_processed.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Main Project Workspace
---
Main script is `pipeline.ipynb`

Output is `videos_out/project_video_processed.mp4`

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `camera_calibrator.py` using the function `get_calibration_matrix()`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

---
### Distortion correction

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

The code for this step is contained in the file called `distortion_correction.py` using the function `get_undistorted_image()`. This function accepts an image, camera matrix as mtx, and distortion coefficients as dist. The function uses `(cv2.undistort(image, mtx, dist, None, mtx))` to undistort the original image, and then returns a undistorted image like the image above.

---
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The code for this step is contained in the file called `image_binary_gradient.py` using the function `get_binary_image(undist_img)`.  Here's an example of my output for this step.  (note: this is from one of the test images)

![alt text][image3]


---

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_trapezoid(img)`, which appears in the file `perspective_transform.py`.  The `get_trapezoid(img)` function takes as inputs an image (`img`), and returns an output as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner from help with the code from [cite source](https://github.com/SlothFriend/CarND-Term1-P4/blob/master/pipeline.ipynb):

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 547.,  480.      | 190.,    0.       | 
| 733.,  480.      | 1090.,    0.      |
| 1090.,  720.     | 1090.,  720.      |
| 190.,  720.      | 190.,  720.       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

---



---
#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for my detect lane pixels and lane boundary includes a function called `find_lane_pixels(image)`, which appears in the file `detect_lane_pixels.py`. The function takes a `top_down` as input (binary thresholded image) that is processed in the `image_binary_gradient.get_binary_image(undist_img)` algorithm. A histogram is drawn at the image base counting the frequency of lane pixels for each column (y-direction). Two peaks are identified, corresponding to the two lanes.

Starting from these two peaks a window rectangle is drawn with a pre-defined height, splitting the height in nine rows. The position of the non-zero (lane pixels) within this window will determine the horizontal (x) position for the following window. Windows are stacked vertically until reaching the top of the image and a quadratic curve is fit through all non-zero pixels (identified as components of the lane) that lie within the window bounds.

The x- and y- values of the non-zero pixels for each lane (left: red and right: blue) are used to determine the coefficients of the polynomial fit. We use a quadratic (2-order) curve. I used the function `fit_polynomial_in_window` from the file `detect_lane_pixels.py` to display the image below that show the lanes found from a sliding window.

![alt text][image5]

Then rather than search the entire next frame for the lines, the output `left_fit, right_fit` from `fit_polynomial_in_window`from previous detection shall be searched using the function `search_around_poly(image)` from the file `detect_lane_pixels.py` that outputs a search area for poly as below images shows.

The polynomial functions are defined as x = f(y) since x can be uniquely defined for each value of y in ploty. The general formulae thus become x = f(y) = Ay2 + By + C with A, B, C corresponding to the values in left_fit and right_fit for each lane.

```python
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    
    img_shape = np.copy(img_shape)
     # Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    # Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty




```

![alt text][image6]

---
---
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in using the function `get_lane_curvature_real(image)` in my file `measure_curvature.py`

We fit a second degree polynomial to the lane lines x = f(y) = Ay2 + By + C. The curvature radius can be calculated as Rcurve = (1 + (2Ay + B)2)3/2 / |2A| . In order to obtain correct curvature values we translate pixel distance to real world coordinates using coefficients. The full image height (720 px) corresponds to roughly 30 m whereas the 3.7 m lane width is accommodated in 700 px width. Corresponding A, B, C coefficients are calculated for the real-world values.

```python

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
```
The curvature is calculated using the real-world coefficients for A and B. The expression is evaluated at the vehicle position which is at 720 px, corresponding to 30 m. Each lane results in its own curvature. The average of the two is used for the visualization.

---

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


I implemented this step in lines `color_zone_warp = measure_curvature.get_color_zone_warp(top_down)`, `newwarp  = perspective_transform.get_original_perspective(color_zone_warp)`, `original_img = cv2.addWeighted(image_original, 1, newwarp, 0.3, 0)`, and `result = measure_curvature.add_text(original_img, top_down)` in my code in `pipeline.py`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I am fairly happy with the results. Lane detection works reasonable well on project_video.mp4 even though, there are frames where the polynomial fit outputs somewhat poorly curvature lanes. These instances can happen when high gradient areas (shadows or marks) are found in the lane which can result in lanes jumping around curve. 

I tried smooting the lane by reducing my region of intrests and color using `clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))` but the results had a slight improvement. 
Performance on more challenging videos is unsatisfactory. In order to achieve better results it will be necessary to implement outlier rejection and use a low-pass filter to smooth the lane detection over frames, meaning add each new detection to a weighted mean of the position of the lines to avoid jitter.

