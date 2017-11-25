import cv2

class PerspectiveTransformer:
    def __init__(self, src, dst):
        self.M_ = cv2.getPerspectiveTransform(src, dst)
        self.M_rev_ = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        return cv2.warpPerspective(img, self.M_, img.shape[1::-1], \
            flags=cv2.INTER_LINEAR)

    def transform_back(self, img):
        return cv2.warpPerspective(img, self.M_rev_, img.shape[1::-1], \
            flags=cv2.INTER_LINEAR)
