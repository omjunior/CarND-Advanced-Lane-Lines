import numpy as np
import cv2


class Lane:
    plot_y = np.linspace(0, 719, num=720)

    def __init__(self):
        self.x = []
        self.y = []
        self.fit = None
        self.lane_inds = None


class LaneFinder:
    def __init__(self, memory):
        self.left = Lane()
        self.right = Lane()
        self.memory = memory
        self.margin = 150
        self.nonzerox = None
        self.nonzeroy = None
        self.success_last = False
        self.limit = 500  # min for failure
        # statistics
        self.stats_full = 0
        self.stats_eq = 0
        self.stats_none = 0

    def __del__(self):
        print("Frames fully analyzed:", self.stats_full)
        print("Frames using equation:", self.stats_eq)
        print("Frames failed:", self.stats_none)

    def find_lanes(self, image):
        if not self.success_last:
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
        return self.success_last

    def mark_lane(self, image):
        left_fitx = self.left.fit[0] * Lane.plot_y ** 2 + self.left.fit[1] * Lane.plot_y + self.left.fit[2]
        right_fitx = self.right.fit[0] * Lane.plot_y ** 2 + self.right.fit[1] * Lane.plot_y + self.right.fit[2]
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, Lane.plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, Lane.plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        color_warp[self.left.y[-1], self.left.x[-1], 0] = 255
        color_warp[self.right.y[-1], self.right.x[-1], 2] = 255

        return color_warp

    def annotate_frame(self, image):
        y_eval = 719

        left_fitx = self.left.fit[0] * Lane.plot_y ** 2 + self.left.fit[1] * Lane.plot_y + self.left.fit[2]
        right_fitx = self.right.fit[0] * Lane.plot_y ** 2 + self.right.fit[1] * Lane.plot_y + self.right.fit[2]
        lanes_dist = right_fitx[y_eval] - left_fitx[y_eval]

        ym_per_pix = 50 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / lanes_dist  # meters per pixel in x dimension

        middlex = (left_fitx[y_eval] + right_fitx[y_eval]) / 2 * xm_per_pix
        carx = 1080 / 2 * xm_per_pix


        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(Lane.plot_y * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(Lane.plot_y * ym_per_pix, right_fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                        np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                         np.absolute(2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        im = cv2.putText(image, "Left curvature: " + "{0:.2f}".format((left_curverad + right_curverad)/2) + "m", (10, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        im = cv2.putText(im, "Car position: " + "{0:.2f}".format(carx - middlex) + "m", (10, 100),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        return im

    def paint(self, image):
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((image, image, image)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left.lane_inds], self.nonzerox[self.left.lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right.lane_inds], self.nonzerox[self.right.lane_inds]] = [0, 0, 255]

        # Generate x and y values for plotting
        left_fitx = self.left.fit[0] * Lane.plot_y ** 2 + self.left.fit[1] * Lane.plot_y + self.left.fit[2]
        right_fitx = self.right.fit[0] * Lane.plot_y ** 2 + self.right.fit[1] * Lane.plot_y + self.right.fit[2]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.margin, Lane.plot_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.margin, Lane.plot_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.margin, Lane.plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.margin, Lane.plot_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        return cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    def find_lanes_full(self, image):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[int(image.shape[0] / 2):, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image.shape[0] / nwindows)
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
        self.left.lane_inds = []
        self.right.lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) &
                              (self.nonzeroy < win_y_high) &
                              (self.nonzerox >= win_xleft_low) &
                              (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) &
                               (self.nonzeroy < win_y_high) &
                               (self.nonzerox >= win_xright_low) &
                               (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.left.lane_inds.append(good_left_inds)
            self.right.lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        self.left.lane_inds = np.concatenate(self.left.lane_inds)
        self.right.lane_inds = np.concatenate(self.right.lane_inds)

        # Extract left and right line pixel positions
        self.left.x.append(self.nonzerox[self.left.lane_inds])
        self.left.y.append(self.nonzeroy[self.left.lane_inds])
        self.right.x.append(self.nonzerox[self.right.lane_inds])
        self.right.y.append(self.nonzeroy[self.right.lane_inds])

        # abort this frame and keep last fit
        if self.right.x[-1].size < self.limit or self.left.x[-1].size < self.limit:
            self.left.x.pop(-1)
            self.left.y.pop(-1)
            self.right.x.pop(-1)
            self.right.y.pop(-1)
            return False

        # hold the last 'memory' items
        if len(self.left.x) > self.memory:
            self.left.x.pop(0)
            self.left.y.pop(0)
            self.right.x.pop(0)
            self.right.y.pop(0)

        # Fit a second order polynomial to each
        flat_ly = np.concatenate(self.left.y).ravel()
        flat_lx = np.concatenate(self.left.x).ravel()
        flat_ry = np.concatenate(self.right.y).ravel()
        flat_rx = np.concatenate(self.right.x).ravel()
        self.left.fit = np.polyfit(flat_ly, flat_lx, 2)
        self.right.fit = np.polyfit(flat_ry, flat_rx, 2)

        return True

    def find_lanes_with_eq(self, image):
        nonzero = image.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        self.left.lane_inds = \
            ((self.nonzerox > (self.left.fit[0] * (self.nonzeroy ** 2) +
                               self.left.fit[1] * self.nonzeroy + self.left.fit[2] - self.margin)) &
             (self.nonzerox < (self.left.fit[0] * (self.nonzeroy ** 2) +
                               self.left.fit[1] * self.nonzeroy + self.left.fit[2] + self.margin)))

        self.right.lane_inds = \
            ((self.nonzerox > (self.right.fit[0] * (self.nonzeroy ** 2) +
                               self.right.fit[1] * self.nonzeroy + self.right.fit[2] - self.margin)) &
             (self.nonzerox < (self.right.fit[0] * (self.nonzeroy ** 2) +
                               self.right.fit[1] * self.nonzeroy + self.right.fit[2] + self.margin)))

        # Again, extract left and right line pixel positions
        self.left.x.append(self.nonzerox[self.left.lane_inds])
        self.left.y.append(self.nonzeroy[self.left.lane_inds])
        self.right.x.append(self.nonzerox[self.right.lane_inds])
        self.right.y.append(self.nonzeroy[self.right.lane_inds])

        # abort this frame and keep last fit
        if self.right.x[-1].size < self.limit or self.left.x[-1].size < self.limit:
            self.left.x.pop(-1)
            self.left.y.pop(-1)
            self.right.x.pop(-1)
            self.right.y.pop(-1)
            return False

        # hold the last 'memory' items
        if len(self.left.x) > self.memory:
            self.left.x.pop(0)
            self.left.y.pop(0)
            self.right.x.pop(0)
            self.right.y.pop(0)

        # Fit a second order polynomial to each
        flat_ly = np.concatenate(self.left.y).ravel()
        flat_lx = np.concatenate(self.left.x).ravel()
        flat_ry = np.concatenate(self.right.y).ravel()
        flat_rx = np.concatenate(self.right.x).ravel()
        self.left.fit = np.polyfit(flat_ly, flat_lx, 2)
        self.right.fit = np.polyfit(flat_ry, flat_rx, 2)

        return True
