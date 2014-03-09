import numpy as np
import cv2
import random

def make_unit_vector(vec):
  return vec / np.linalg.norm(vec)

def angle_between_lines(line1, line2):
  x1, x2, y1, y2 = line1
  vec1 = np.array([x2 - x1, y2 - y1])

  x1, x2, y1, y2 = line2
  vec2 = np.array([x2 - x1, y2 - y1])

  vec1 = make_unit_vector(vec1)
  vec2 = make_unit_vector(vec2)

  angle = np.arccos(np.dot(vec1, vec2))
  if np.isnan(angle):
    return 0.0

  if angle > np.pi / 2:
    angle = np.pi - angle
  return angle * float(180) / np.pi

image = cv2.imread('../small-images/1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(image_gray, 50, 200)
polar_lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)[0]
lines = [];

cv2.imshow('edges', edges)

for i, line in enumerate(polar_lines):
  r, theta = line
  rise = -np.cos(theta)
  run = np.sin(theta)

  x0 = r * np.cos(theta)
  y0 = r * np.sin(theta)

  x1 = int(x0 + run * -1000)
  y1 = int(y0 + rise * -1000)
  x2 = int(x0 + run * 1000)
  y2 = int(y0 + rise * 1000)

  lines.append([x1, x2, y1, y2])

best_lines = []

for i in range(100):
  sampled_lines = random.sample(lines, 4)
  sampled_angles = np.array([
    angle_between_lines(sampled_lines[0], sampled_lines[1]),
    angle_between_lines(sampled_lines[0], sampled_lines[2]),
    angle_between_lines(sampled_lines[0], sampled_lines[3]),
  ])

  print np.sort(sampled_angles)

  if sampled_angles[0] < 5 and sampled_angles[1] > 85 and sampled_angles[2] > 85:
    best_lines.append(sampled_lines)

print best_lines[0]
for line in best_lines[0]:
  x1, x2, y1, y2 = line
  cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 2)
cv2.imshow('hough', image)


while True:
  # handle events
  k = cv2.waitKey(10)

  if k == 0x1b: # quit on esc
    print 'ESC pressed. Exiting ...'
    break


