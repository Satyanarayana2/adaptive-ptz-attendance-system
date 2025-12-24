# utils/face_alignment.py

import numpy as np
import cv2

# Standard 5-point template for ArcFace alignment
ARC_TEMPLATE_5PT = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041]   # right mouth
], dtype=np.float32)

class FaceAligner:

    def __init__(self, output_size=(112, 112)):
        self.output_size = output_size
        self.template = ARC_TEMPLATE_5PT

    def align(self, frame, kps):
        """
        frame: original frame (BGR)
        kps: dict with 5 keys: 'left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth'

        Returns: aligned 112x112 BGR face crop
        """

        frame = self.denoise(frame)
        try:
            src = np.array([
                kps["left_eye"],
                kps["right_eye"],
                kps["nose"],
                kps["left_mouth"],
                kps["right_mouth"]
            ], dtype=np.float32)
        except Exception as e:
            print("Keypoints format invalid:", e)
            return None

        # Estimate similarity transform
        transform_matrix, _ = cv2.estimateAffinePartial2D(src, self.template, method=cv2.LMEDS)

        if transform_matrix is None:
            return None

        aligned_face = cv2.warpAffine(
            frame, 
            transform_matrix,
            self.output_size,
            borderValue=0.0
        )
        aligned_face = self.normalize_lighting(aligned_face)
        return aligned_face

    def normalize_lighting(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def denoise(self, img):
        return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)