# Advanced Lane Finding Project

## Goals

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[cal_dist]: ./camera_cal/calibration2.jpg
[cal_undist]: ./output_images/camera_cal/calibration2.jpg
[t1_dist]: ./test_images/straight_lines1.jpg
[t2_dist]: ./test_images/test2.jpg
[t1_undist]: ./output_images/01-undistort/straight_lines1.jpg
[t2_undist]: ./output_images/01-undistort/test2.jpg
[t1_bin]: ./output_images/02-binary/straight_lines1.jpg
[t2_bin]: ./output_images/02-binary/test2.jpg
[t1_warp_orig1]: ./output_images/03.1-warp/original/straight_lines1.jpg
[t2_warp_orig1]: ./output_images/03.1-warp/original/test2.jpg
[t1_warp_orig2]: ./output_images/03.1-warp/warped-original/straight_lines1.jpg
[t2_warp_orig2]: ./output_images/03.1-warp/warped-original/test2.jpg
[t1_warp_bin]: ./output_images/03.1-warp/warped-binary/straight_lines1.jpg
[t2_warp_bin]: ./output_images/03.1-warp/warped-binary/test2.jpg
[t1_lane]: ./output_images/04-lanes/straight_lines1.jpg
[t2_lane]: ./output_images/04-lanes/test2.jpg
[t1_unwarped]: ./output_images/05-unwarped/straight_lines1.jpg
[t2_unwarped]: ./output_images/05-unwarped/test2.jpg
[t1_final]: ./output_images/straight_lines1.jpg
[t2_final]: ./output_images/test2.jpg

## Rubric Points

### Camera Calibration

The code for the camera calibration was implemented on the class ```Camera```.
The method ```findChessboardCorners``` from OpenCV is executed for each image taken from the camera, which finds the
positions of each crossing in a chessboard image photo, accumulated in ```imgpoints```.
The points are known to be in a plane, so the reference points (```objpoints```), which are points in 3D, are considered
flat (z coordinate always zero).
Both ```objpoints``` and ```imgpoints``` are passed to the ```calibrateCamera``` OpenCV function to compute the
camera matrix and distortion coefficients, which in turn are used to undistort images taken with the same camera.

Follow an example of these calibration images, and the result of undistorting it.

| Original distorted image | Undistorted image |
|:---:|:---:|
| ![alt text][cal_dist] | ![alt_text][cal_undist] |

### Pipeline (single images)

The file ```images.py``` calls the following for each image in ```./test_image```.

#### Remove camera distortion

The first step of the pipeline is to remove camera distoritions, using the matrix and coefficients found earlier.
The distortions in this particular case aren't great, being most noticeable in the border of the pictures.

Follow examples.

| Original distorted image | Undistorted image |
|:---:|:---:|
| ![alt text][t1_dist] | ![alt_text][t1_undist] |
| ![alt text][t2_dist] | ![alt_text][t2_undist] |

#### Create a binary image identifying the lanes

I found that the most efficient method for identifying pixels belonging to lanes was not applying convolution, but
simple thresholds.

I converted the image to the HLS space, and combined a yellow thresholding with a white one.
The yellow threshold was implemented by seraching for a Hue channel value between 0 and 50, and both Luminance and
Saturation between 100 and 255. As for the white threshold, any pixel with Luminance larger than 200 is accepted.

This was implemented in the class ```ImagePipeline```, method ```find_lanes()```, which should receive an image already
in HLS colorspace, and is as simple as:
```python
    def find_lanes(hls):
        mask_yellow = cv2.inRange(hls, np.array([0, 100, 100]), np.array([50, 255, 255]))
        mask_white = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
        return np.logical_or(mask_yellow, mask_white)
```

The results are as follow:

| | | 
|:---:|:---:|
| ![alt text][t1_bin] | ![alt_text][t2_bin] |


#### Perspective transform

The perspective transform code is divided in two classes: ```AOIBuilder``` and ```PerspectiveTransformer```.

The first is a helper class used to define the mapping between point in the original and warped views.
Instead of defining fixed points, I found easier to give relative percentages of the whole screen.

It's functionality can be see on the constructor:

```python
    def __init__(self, width, height, bottom_width, bottom_y_pos, top_width, top_y_pos):
        self.w_ = width
        self.h_ = height
        self.src_ = \
            [[self.w_ / 2 - (self.w_ * bottom_width) / 2, self.h_ * bottom_y_pos],
             [self.w_ / 2 + (self.w_ * bottom_width) / 2, self.h_ * bottom_y_pos],
             [self.w_ / 2 - (self.w_ * top_width) / 2, self.h_ * top_y_pos],
             [self.w_ / 2 + (self.w_ * top_width) / 2, self.h_ * top_y_pos]]
```

Then the class ```PerspectiveTranformer``` computes and stores both the direct and reverse matrices using OpenCV's 
```getPerspectiveTransform```, and calls OpenCV's ```warpPerspective``` when needed with the rigth matrix.

To better illustrate, a red box is drawn over the original image, so the process can be understood:

| Original marked | Warped region | 
|:---:|:---:|
| ![alt text][t1_warp_orig1] | ![alt_text][t1_warp_orig2] |
| ![alt text][t2_warp_orig1] | ![alt_text][t2_warp_orig2] |

On the actual pipeline, the image transformed is the binary one. Follow an example:

| |  | 
|:---:|:---:|
| ![alt text][t1_warp_bin] | ![alt_text][t2_warp_bin] |


#### Find Lines

The code for finding 2nd order polinomials can be found in the class ```LaneFinder```.

There are two methods for finding pixels, one for starting from scratch, called ```find_lanes_full()```, and another
for when the equation for the last frame is known and reliable, called ```find_lanes_with_eq()```.

The first method, ```find_lanes_full()```, starts by looking at the bottom half of the binary thresholded image, 
computing the histogram of pixels by column. The two larger peaks are chosen as the starting points for the search as
right and left lane points.

..............................................................................

In both methods, after the points belonging to the lanes are found, they are stored in a list with at most ```memory```
positions, and these points are used in the polinomial regression. Unless a minimum number of points is not found. In
this case the search is considered a failure, and the points are not used. 

This implements a sort of low pass filtering, since the regression algorithm will see points from the last ```memory```
frames. This was done so the generated video would be less wobbly. I found that low values work best (3 to 5 frames).


In the following examples, the detected points belonging to lines are colored blue and red, and the space between the
computed polinomials (the lane) is colored green:

| |  | 
|:---:|:---:|
| ![alt text][t1_lane] | ![alt_text][t2_lane] |
 

Which after being warped back and overlaid on the original (undistorted) image, produced the following:

| |  | 
|:---:|:---:|
| ![alt text][t1_unwarped] | ![alt_text][t2_unwarped] |
 

#### Curvature computation

| |  | 
|:---:|:---:|
| ![alt text][t1_final] | ![alt_text][t2_final] |

---

### Pipeline (video)

The same pipeline is applied to all videos in the root directory in the file ```video.py```.

The output for the project video can be found [in this link](./output_video/project_video.mp4)

---

### Discussion

