import cv2

class PerspectiveTransformer:
    """ Perspective transformation class """
    def __init__(self, src, dst):
        """ Constructor """
        self.M_ = cv2.getPerspectiveTransform(src, dst)
        self.M_rev_ = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        """ Transform from src coordinates to dst """
        return cv2.warpPerspective(img, self.M_, img.shape[1::-1])

    def transform_back(self, img):
        """ Transform from dst coordinates to src """
        return cv2.warpPerspective(img, self.M_rev_, img.shape[1::-1])

class AOIBuilder:
    """ Class to define Area of Interest """
    def __init__(self, width, height, \
            bottom_width, bottom_y_pos, top_width, top_y_pos):
        """ Constructor """
        self.w_ = width
        self.h_ = height
        self.src_ = \
            [[ self.w_/2 - (self.w_*bottom_width)/2, self.h_*bottom_y_pos ], \
             [ self.w_/2 + (self.w_*bottom_width)/2, self.h_*bottom_y_pos ], \
             [ self.w_/2 - (self.w_*top_width)/2, self.h_*top_y_pos ], \
             [ self.w_/2 + (self.w_*top_width)/2, self.h_*top_y_pos ]]

    def get_src_points(self):
        """ Return the AOI in src coordinates """
        return self.src_

    def get_dst_points(self):
        """ Return the AOI in dst coordinates """
        return [[0, self.h_], [self.w_, self.h_], [0, 0], [self.w_, 0]]

    def get_lines(self):
        """ Return the lines that define AOI in src coordinates """
        return [[self.src_[0][0], self.src_[0][1], self.src_[1][0], self.src_[1][1]], \
                [self.src_[1][0], self.src_[1][1], self.src_[3][0], self.src_[3][1]], \
                [self.src_[3][0], self.src_[3][1], self.src_[2][0], self.src_[2][1]], \
                [self.src_[2][0], self.src_[2][1], self.src_[0][0], self.src_[0][1]]]
