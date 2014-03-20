import numpy as np
import cv2
import classify

image = cv2.imread('adam.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
splits = range(0, image.shape[1], 20)

for split in splits:
  cv2.line(image, (split, 0), (split, image.shape[0]), (255, 0, 0))
cv2.imshow('adam', image)

start = splits[8]
for split in splits[10:25]:
  sub_image = image_gray[70 : image.shape[0], start : split]
  sub_image = cv2.GaussianBlur(sub_image, (5, 5), 8)

  value, sub_image = cv2.threshold(sub_image, 200, 255, cv2.THRESH_BINARY)
  cv2.imshow('sub_image', sub_image)

  features = classify.compute_features(sub_image)


classifier = classify.read_classifier()
print classifier.predict(features)
print classifier.predict_proba(features)

while True:
  # handle events
  k = cv2.waitKey(10)

  if k == 0x1b: # quit on esc
    print 'ESC pressed. Exiting ...'
    break
