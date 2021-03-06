import numpy as np
import cv2
import glob
import pickle
import os.path


class Camera:
    """ Camera class to compute camera parameters and undistort images """

    def __init__(self, pickle_file):
        """
        Constructor

        If a pickle file is given, just loads mtx and dist
        If no pickle file, then compute data
        """
        if os.path.exists(pickle_file):
            # load pickle
            data = pickle.load(open(pickle_file, mode='rb'))
            self.mtx = data["mtx"]
            self.dist = data["dist"]
        else:
            objp = np.zeros((9 * 6, 3), np.float32)
            objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 3d points
            objpoints = []
            imgpoints = []
            images = glob.glob('camera_cal/calibration*.jpg')
            for idx, fname in enumerate(images):
                img = cv2.imread(fname)
                img_size = (img.shape[1], img.shape[0])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)
            _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
            self.mtx = mtx
            self.dist = dist
            # save pickle
            dist_pickle = {"mtx": self.mtx, "dist": self.dist}
            pickle.dump(dist_pickle, open(pickle_file, "wb"))

    def undistort(self, img):
        """ Undistort an image """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
