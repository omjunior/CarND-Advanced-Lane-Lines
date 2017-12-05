import cv2
import numpy as np
from camera import Camera
from perspective import PerspectiveTransformer, AOIBuilder


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    """ Draw lines over an image """
    for line in lines:
        x1,y1,x2,y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def abs_sobel_thresh(img, orient, sobel_kernel, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        return None
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary

def abs_sobel_adapthresh(img, orient, sobel_kernel, adap_size, adap_c):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        return None
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = cv2.adaptiveThreshold(scaled_sobel, 255, \
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adap_size, adap_c)
    return binary

def mag_thresh(img, sobel_kernel, mag_thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2+sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary

def mag_adapthresh(img, sobel_kernel, adap_size, adap_c):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2+sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = cv2.adaptiveThreshold(scaled_sobel, 255, \
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adap_size, adap_c)
    return binary

def dir_thresh(img, sobel_kernel, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    binary = np.zeros_like(dir_sobel)
    binary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    return binary

def hls_s_thresh(img, thresh):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    _, _, s = cv2.split(hls)
    binary = np.zeros_like(s)
    binary[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary

def hls_s_adapthresh(img, adap_size, adap_c):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    _, _, s = cv2.split(hls)
    binary = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        cv2.THRESH_BINARY, adap_size, adap_c)
    return binary

class ImagePipeline:
    def __init__(self):
        self.cam_ = Camera("camera_cal/calibration.p")
        self.aoi_ = AOIBuilder(1280, 720, 1.0, 0.95, 0.26, 0.68)
        self.persp_ = PerspectiveTransformer( \
            np.array(self.aoi_.get_src_points(), dtype=np.float32), \
            np.array(self.aoi_.get_dst_points(), dtype=np.float32))

    def find_lanes(self, hls):
        # detect yellow by hue (and not so low ligthness and saturation)
        mask_yellow = cv2.inRange(hls, np.array([0,100,100]), np.array([50,255,255]))
        # return np.array(mask_yellow, dtype=np.bool_)
        # detect white by high lightness
        mask_white = cv2.inRange(hls, np.array([0,200,0]), np.array([255,255,255]))
        # return np.array(mask_white, dtype=np.bool_)
        return np.logical_or(mask_yellow, mask_white)

    def undistort(self, img):
        # undistort image
        und = self.cam_.undistort(img)
        return und

    def process_frame(self, img):
        # convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # find lanes
        mask = self.find_lanes(hls)
        mask = np.array(mask, dtype=np.uint8)
        # return np.dstack(( mask, mask, mask)) * 255
        # warp perspective
        warped = self.persp_.transform(np.array(mask, dtype=np.uint8))
        #return np.dstack(( warped, warped, warped)) * 255
        return warped

    def warp_back(self, img, mask):
        unwarped_mask = self.persp_.transform_back(mask)
        comb = cv2.addWeighted(img, 1, unwarped_mask, 0.3, 0)
        return comb
