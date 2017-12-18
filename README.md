# Advanced Lane Finding Project

## Goals

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
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
The points are known to be on a plane, so the reference points (```objpoints```), which are points in 3D, are considered
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

The first step of the pipeline is to remove camera distortions, using the matrix and coefficients found earlier.
The distortions in this particular case aren't great, being most noticeable on the border of the pictures.

Follow examples.

| Original distorted image | Undistorted image |
|:---:|:---:|
| ![alt text][t1_dist] | ![alt_text][t1_undist] |
| ![alt text][t2_dist] | ![alt_text][t2_undist] |

#### Create a binary image identifying the lanes

I found that the most efficient method for identifying pixels belonging to lanes was not applying convolutional filters,
but to apply simple threshold values.

I converted the image to the HLS space and combined a yellow thresholding with a white one.
The yellow threshold was implemented by searching for a Hue channel value between 0 and 50, and both Luminance and
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

The perspective transform code is divided into two classes: ```AOIBuilder``` and ```PerspectiveTransformer```.

The first is a helper class used to define the mapping between a point in the original and warped views.
Instead of defining fixed points, I found easier to give relative percentages of the whole screen.

Its functionality can be seen on the constructor:

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
```getPerspectiveTransform```, and calls OpenCV's ```warpPerspective``` when needed with the right matrix.

In this work, the parameters found to represent well the dataset were ```(1280, 720, 1.2, 0.96, 0.12, 0.62)```, which
produce the following translation: 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| -128, 691     |    0, 720     | 
| 1408, 691     | 1280, 720     |
|  563, 446     |    0, 0       |
|  716, 446     | 1280, 0       |

Notice that the bottom corners are actually outside the image, which explains the black portions of the warped image.
This was done to be able to have a greater lateral extension on the top of the region of interest while keeping the
lines 'straight'.

To better illustrate, a red box is drawn over the original image, so the process can be better understood:

| Original marked | Warped region | 
|:---:|:---:|
| ![alt text][t1_warp_orig1] | ![alt_text][t1_warp_orig2] |
| ![alt text][t2_warp_orig1] | ![alt_text][t2_warp_orig2] |

On the actual pipeline, the image transformed is the binary one. Follow an example:

| |  | 
|:---:|:---:|
| ![alt text][t1_warp_bin] | ![alt_text][t2_warp_bin] |


#### Find Lines

The code for finding 2nd order polynomials can be found in the class ```LaneFinder```.

There are two methods for finding pixels, one for starting from scratch, called ```find_lanes_full()```, and another
for when the equation for the last frame is known and reliable, called ```find_lanes_with_eq()```.

The first method, ```find_lanes_full()```, starts by looking at the bottom half of the binary thresholded image, 
computing the histogram of 'on' pixels by column. The larger peak of each half of the image is computed.

Then search blocks are defined. The height of each block is ```1/nwindows``` of the frame height (in this case
```nwindows``` used was 9). The first blocks for both left and right are centered in the respective peaks found earlier.
Each block extends laterally by ```margin``` pixels (150 used in this case).
All non-zero pixels inside these blocks are selected. The average of the **x** positions of each block will determine
the center of the next search block.

The method ```find_lanes_with_eq()``` also finds the pixels belonging to the lines, but with a different approach.
Instead of defining blocks, the pixels selected are those within plus or minus ```margin``` from the previous equation
found earlier. That is, for each **y**, the corresponding **x** value is computed by applying the polynomial found in
the previous frame. Every non-zero pixel for that **y**, in a distance less than ```margin``` from the computed **x**
are selected.

For both cases, if a minimum of ```limit``` points are not found for either line, the result is rejected, and they are
not used to compute a new polynomial.

In both methods, after the points belonging to the lanes are found, they are stored in a list with at most ```memory```
positions, and these points are used in the polynomial regression.
This implements a sort of low pass filtering since the regression algorithm will see points from the last ```memory```
frames. This was done so the generated video would be less wobbly. I found that low values work best (3 to 5 frames).

If the last frame failed, the current frame will be processed with ```find_lanes_full()```. If the last frame was
successful, the current frame is processed with ```find_lanes_with_eq()```, and if this method fails the same frame
falls to ```find_lanes_full()```.

The polynomials are computed by using the OpenCV function ```polyfit``` over the selected points.

In the following examples, the detected points belonging to the left and right lines are colored blue and red
respectively, and the space between the computed polynomials (the lane) is colored green:

| |  | 
|:---:|:---:|
| ![alt text][t1_lane] | ![alt_text][t2_lane] |
 

Which after being warped back and overlaid on the original (undistorted) image, produced the following:

| |  | 
|:---:|:---:|
| ![alt text][t1_unwarped] | ![alt_text][t2_unwarped] |
 

#### Curvature computation

Finally, the radius of curvature of the lane and relative position of the car are calculated.

At this point, we have a polynomial representing each of the lines delimiting the lane.
These equations are of the form: ```x(y) = A * y^2 + B * y + C```.
The regression is taken as **x** being a function of **y** rather than the opposite because a function can only have one
output value for each input, and there can be lots of **y** values for the same **x** since lines can be very close to
vertical.

The curvature of a line at a point **y** can be computed using the following formula:
 
```R = ((1 + (2*y+B)^2)^(3/2))/abs(2*A)```

But since all these points are being measured in pixels, a conversion must be performed to express these values in
real-world units, such as meters, prior to compute the radius of curvature.

For that, the equation is first computed for each **y** in the range of the image (its height, 720), defining one
```(x, y)``` point with **y** varying from 0 to 719.
For the **x** dimension, the distance between lines is computed in pixels, and the following constant is multiplied
to each **x**: ```3.7 meters / lanes_dist pixels```, assuming the width of the lane to be 3.7 meters.
For the **y** dimension, it is assumed that the area of interest selected spans approx. 50 meters, hence multiplying
by a factor of ```50 meters / 720 pixels```.

After the transformation, a new regression is performed for each of the lines, and the above equation is applied to the
bottom-most pixel (the point closest to the car). The average of the curvatures of the right and left lanes is taken.

For the car's position, it is assumed that the camera is mounted on the center of the car, so the center of the image
is a good representative to the position of the car. So the position is computed as the difference between the center
of the image to the middle point of the lanes ```(right_lane - left_lane)```.

The computed values are annotated over the image, as follows:

| |  | 
|:---:|:---:|
| ![alt text][t1_final] | ![alt_text][t2_final] |

### Pipeline (video)

The same pipeline is applied to all videos in the root directory in the file ```video.py```.

The output for the project video can be found [at this link](./output_video/project_video.mp4)

In this particular video, only the first frame was fully computed using ```find_lanes_full()```.
All other frames were computed by using ```find_frames_with_eq()```.

---

### Discussion

First of all the detection of lines relies on them being either white or yellow. Any lane defined by lines of a
different color would certainly make the algorithm fail.

The constants used were also fine-tuned for this video, and so might not
generalize well. For this issue, maybe a pre-processing with a sliding window performing an adaptive equalization could
help to make the colors a little bit less variable.

The algorithm also always looks for two lines, one on the right side and one on the left side of the frame.
So whenever the car changes lanes there would be an interval where it can't detect the lane.

A good diagnostics for finding were the algorithm is underperforming would be to create a video output with the screen
split in four, showing the binary image, the bird's eye view of the binary image, the detected line points and the
output frame all toghether.