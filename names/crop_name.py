import cv2
import os
import math
import numpy as np
from PIL import Image
import pdb
import random

class CropName:

    def __init__(self, directory):
        self.directory = directory
        self.NUM_ITERATIONS = 200
        self.THRESHOLD = 5.0

        # Variables to describe the bounding box around the name for the blank exam
        self.blank_left = 320
        self.blank_top = 380
        self.blank_height = 100
        self.blank_width = 800

        self.detector = cv2.FeatureDetector_create('ORB')
        self.descriptor = cv2.DescriptorExtractor_create('SIFT')
        self.matcher = cv2.DescriptorMatcher_create('FlannBased')


    def run(self):
        blank_image = cv2.imread(os.path.join(self.directory, 'blank0.jpeg'))
        self.size = (blank_image.shape[1], blank_image.shape[0])
        print 'Computing keypoints for the blank page...'
        keypoints = self.detector.detect(blank_image)
        # pdb.set_trace()
        blank_kp, blank_descriptor = self.descriptor.compute(blank_image, keypoints)

        cropped_images = []
        names = []
        for filename in os.listdir(self.directory):
            if 'blank' in filename:
                continue
            student_image = cv2.imread(os.path.join(self.directory, filename))
            # Resize the images to be the same size
            student_image = cv2.resize(student_image, self.size)

            transform = self._get_transformation(blank_image, keypoints, blank_kp, blank_descriptor, student_image)

            transform = np.append(transform, [0, 0, 1])
            transform.resize(3, 3)
            inverse = np.linalg.inv(transform)
            transformed_student_image = cv2.warpAffine(student_image, inverse[0:2, 0:3], self.size)

            y1 = self.blank_top
            y2 = self.blank_top + self.blank_height
            x1 = self.blank_left
            x2 = self.blank_left + self.blank_width
            cv2.imwrite('testtest-upstream.jpeg', transformed_student_image)
            cropped_image = transformed_student_image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

            filename = filename.replace('-', ' ').replace('.jpeg', '')
            index = max([i for i in range(len(filename)) if filename[i].isalpha()])
            filename = filename[0:index + 1]
            names.append(filename)

        return cropped_images, names


    def _get_transformation(self, blank_image, keypoints, blank_kp, blank_descriptor, student_image):
        print 'Computing keypoints for the student page...'
        keypoints = self.detector.detect(student_image)
        page_kp, page_descriptor = self.descriptor.compute(student_image, keypoints)

        print 'Matching keypoints...'
        matches = self.matcher.match(blank_descriptor, page_descriptor)
        # pdb.set_trace()

        print 'Running RANSAC...'
        best_transform = None
        best_num_outliers = 10000
        for i in range(self.NUM_ITERATIONS):
            if i < 10:
                cur_matches = random.sample(sorted(matches, key=lambda match: match.distance)[0:20], 4)
            else:
                cur_matches = random.sample(matches, 4)
            blank_points = np.array(map(lambda match: [blank_kp[match.queryIdx].pt], cur_matches))
            page_points = np.array(map(lambda match: [page_kp[match.trainIdx].pt], cur_matches))
            transform = cv2.estimateRigidTransform(np.float32(blank_points), np.float32(page_points), fullAffine=False)
            if transform is None:
                continue

            num_outliers = 0
            num_inliers = 0
            for match in matches:
                blank_point = np.matrix([blank_kp[match.queryIdx].pt[0], blank_kp[match.queryIdx].pt[1], 1])
                blank_point = blank_point.reshape(3, 1)
                estimated_page_point = np.asmatrix(transform) * blank_point
                actual_page_point = page_kp[match.trainIdx].pt
                euclidean_distance = math.sqrt((estimated_page_point[0] - actual_page_point[0])**2 +
                    (estimated_page_point[1] - actual_page_point[1])**2)
                if euclidean_distance > self.THRESHOLD:
                    num_outliers += 1
                else:
                    num_inliers += 1
                if num_outliers >= best_num_outliers:
                    break
            if num_outliers < best_num_outliers:
                best_transform = transform
                best_num_outliers = num_outliers

        return best_transform

    # cv2.imwrite('transformed.jpeg', transformed_student_image)
    # cv2.imwrite('cropped-transformed.jpeg', cropped_name)
    # cropped_blank = blank_image[top:top + height, left:left + width]
    # cv2.imwrite('cropped-blank.jpeg', cropped_blank)

