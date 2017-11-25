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
        return cv2.warpPerspective(img, self.M_rev_, img.shape[1::-1], \
            flags=cv2.INTER_LINEAR)
