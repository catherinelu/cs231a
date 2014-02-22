import numpy as np
import cv2
import pdb

images = ['small-images/1.jpg', 'small-images/2.jpg']
images = map(lambda image: cv2.imread(image, 0), images)

transform = cv2.estimateRigidTransform(images[0], images[1], fullAffine=False)
transform[0:2, 0:2] /= ((transform[0][0] ** 2 + transform[0][1] ** 2) ** 0.5)

print images[0].dtype
