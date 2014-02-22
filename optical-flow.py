import numpy as np
import cv2
import pdb

images = ['small-images/1.jpg', 'small-images/2.jpg']
images = map(lambda image: cv2.imread(image, 0), images)

features = cv2.goodFeaturesToTrack(images[0], maxCorners=100, qualityLevel=0.3,
  minDistance=7, blockSize=7)
next_points, statuses, errors = cv2.calcOpticalFlowPyrLK(images[0], images[1],
  features)

# create a mask for drawing
features_mask = np.zeros_like(images[0])

# select the points which were mapped well
original_points = features[statuses == 1]
moved_points = next_points[statuses == 1]

# for (original, moved) in zip(original_points, moved_points):
#   original_x, original_y = original.ravel()
#   moved_x, moved_y = moved.ravel()
# 
#   cv2.line(features_mask, (original_x, original_y), (moved_x, moved_y), (255, 255, 255), 2)
#   cv2.circle(images[0], (original_x, original_y), 5, (51, 102, 187), -1)

original_points = original_points.reshape(-1, 1, 2)
moved_points = moved_points.reshape(-1, 1, 2)

transform = cv2.estimateRigidTransform(original_points, moved_points, fullAffine=False)
height, width = images[0].shape
transformed = cv2.warpAffine(images[0], transform, (width, height))

cv2.imshow('second_image', images[1])
cv2.imshow('transformed_first_image', transformed)

transform[0:2, 0:2] /= ((transform[0][0] ** 2 + transform[0][1] ** 2) ** 0.5)
print transform

# print cv2.estimateRigidTransform(images[0], images[1], fullAffine=True)
# 
# # get rid of scaling
# scaling_factor = transform[0][0] ** 2 + transform[0][1] ** 2
# for i in range(0, 2):
#   transform[0][i] = transform[0][i] / scaling_factor
#   transform[1][i] = transform[1][i] / scaling_factor

# first_image_with_features = cv2.add(images[0], features_mask)
# cv2.imshow('first_image_with_features', first_image_with_features)
# cv2.imshow('second_image', images[1])

while True:
  # handle events
  k = cv2.waitKey(10)

  if k == 0x1b: # quit on esc
    print 'ESC pressed. Exiting ...'
    break
