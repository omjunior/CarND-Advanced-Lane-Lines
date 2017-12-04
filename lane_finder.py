import numpy as np
import cv2
import matplotlib.pyplot as plt


class LaneFinder:
    def __init__(self, width, memory):
        self.left = None
        self.right = None
        self.memory = memory
        self.success_last = False
        self.default_left = [0,0,200]
        self.default_right = [0,0,1000]
        self.margin = 100
        # to consider a failure
        self.limit = 1000
        # statistics
        self.stats_full = 0
        self.stats_eq = 0
        self.stats_none = 0
        # save to draw later

    def __del__(self):
        print("Frames fully analyzed:", self.stats_full)
        print("Frames using equation:", self.stats_eq)
        print("Frames failed:", self.stats_none)

    def find_lanes(self, image):
        if (not self.success_last):
            self.success_last = self.find_lanes_full(image)
            if self.success_last:
                self.stats_full += 1
            else:
                self.stats_none += 1
        else:
            self.success_last = self.find_lanes_with_eq(image)
            if self.success_last:
                self.stats_eq += 1
            else:
                self.success_last = self.find_lanes_full(image)
                if self.success_last:
                    self.stats_full += 1
                else:
                    self.stats_none += 1
        return self.success_last, self.left, self.right

    def paint(self, image):
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((image, image, image))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fitx = self.left[0]*ploty**2 + self.left[1]*ploty + self.left[2]
        right_fitx = self.right[0]*ploty**2 + self.right[1]*ploty + self.right[2]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin,
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin,
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        return cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    def find_lanes_full(self, image):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[int(image.shape[0]/2):,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
            (self.nonzerox >= win_xleft_low) &  (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
            (self.nonzerox >= win_xright_low) &  (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        # Extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        #print(len(leftx), len(rightx))
        if (len(rightx) < self.limit or len(leftx) < self.limit):
            self.left = self.default_left
            self.right = self.default_right
            return False

        # Fit a second order polynomial to each
        self.left = np.polyfit(lefty, leftx, 2)
        self.right = np.polyfit(righty, rightx, 2)

        return True


    def find_lanes_with_eq(self, image):
        nonzero = image.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        self.left_lane_inds = ((self.nonzerox > (self.left[0]*(self.nonzeroy**2) + self.left[1]*self.nonzeroy +
        self.left[2] - self.margin)) & (self.nonzerox < (self.left[0]*(self.nonzeroy**2) +
        self.left[1]*self.nonzeroy + self.left[2] + self.margin)))

        self.right_lane_inds = ((self.nonzerox > (self.right[0]*(self.nonzeroy**2) + self.right[1]*self.nonzeroy +
        self.right[2] - self.margin)) & (self.nonzerox < (self.right[0]*(self.nonzeroy**2) +
        self.right[1]*self.nonzeroy + self.right[2] + self.margin)))

        # Again, extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        #print(len(leftx), len(rightx))
        if (len(rightx) < self.limit or len(leftx) < self.limit):
            self.left = self.default_left
            self.right = self.default_right
            return False

        # Fit a second order polynomial to each
        self.left = np.polyfit(lefty, leftx, 2)
        self.right = np.polyfit(righty, rightx, 2)

        return True
