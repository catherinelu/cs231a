from collections import defaultdict
import cv2
import fnmatch
import math
import numpy as np
import os
from PIL import Image, ImageDraw
import pdb
import sys
import random
import utils



class Segmenter:
    """ Given a file, outputs the segmented characters for it. """

    def __init__(self, filename):
        self.img = cv2.imread(filename, 0)

        self.WHITE_PIXEL = 255

        self.WHITE_THRESHOLD = 250
        self.NUM_CHAR_THRESHOLD = 20
        self.NUM_POINTS_FOR_LINE_THRESHOLD = 20
        self.DISTANCE_TO_LINE_THRESHOLD = 5.0

        self.MAX_WIDTH = 400
        self.MAX_HEIGHT = 80

        sys.setrecursionlimit(1500)


    def segment(self):
        """
        Returns list of lists of (x, y) coordinates. The list of (x, y)
        coordinates each represent one segmented character.
        """
        height, width = self.img.shape

        if width > self.MAX_WIDTH:
            width = self.MAX_WIDTH
            height = int(self.MAX_WIDTH / float(width) * height)
        if height > self.MAX_HEIGHT:
            height = self.MAX_HEIGHT
            width = int(self.MAX_HEIGHT / float(height) * width)

        self.img = cv2.resize(self.img, (width, height))

        # First, put the indices of all the non-white pixels into a set
        non_white_pixels = set()
        for x in range(width):
            for y in range(height):
                if self.img[y, x] < self.WHITE_THRESHOLD:
                    non_white_pixels.add((x, y))

        # Next, cluster pixels into groups
        self.characters = []  # List of lists of character groupings
        while len(non_white_pixels) > 0:
            non_white_pixel = non_white_pixels.pop()
            group = [non_white_pixel]

            old_x = non_white_pixel[0]
            old_y = non_white_pixel[1]

            for x in xrange(-1, 2):
                for y in xrange(-1, 2):
                    if (x == 0) != (y == 0):  # XOR
                        self._create_cluster(old_x + x, old_y + y, group, non_white_pixels, width, height)
            self.characters.append(group)

        # Get rid of characters that do not have enough pixels (likely to not be an actual character)
        self._remove_noise()

        # Sort the characters based on the smallest x, y coordinate (which is smallest x)
        self.characters.sort()

        # Possibly get rid of the name line.
        self._try_to_remove_line()

        return self.characters


    def _create_cluster(self, cur_x, cur_y, group, non_white_pixels, width, height):
        if cur_x < 0 or cur_x >= width or cur_y < 0 or cur_y >= height:
            return
        if (cur_x, cur_y) in non_white_pixels:
            group.append((cur_x, cur_y))
            non_white_pixels.remove((cur_x, cur_y))
            for x in xrange(-1, 2):
                for y in xrange(-1, 2):
                    if (x == 0) != (y == 0):  # XOR
                        self._create_cluster(cur_x + x, cur_y + y, group, non_white_pixels, width, height)


    def _remove_noise(self):
        significant_characters = []
        for character in self.characters:
            if len(character) > self.NUM_CHAR_THRESHOLD:
                significant_characters.append(character)
        self.characters = significant_characters    


    def _try_to_remove_line(self):
        edges = cv2.Canny(self.img, 300, 500)
        # 50 = Hough threshold
        # Detect lines
        polar_lines = [l[0] for l in cv2.HoughLines(edges, 1, np.pi / 180, 200)]
        lines = [];

        for i, line in enumerate(polar_lines):
            r, theta = line
            rise = -np.cos(theta)
            run = np.sin(theta)

            x0 = r * np.cos(theta)
            y0 = r * np.sin(theta)

            if run == 0:
                run = 0.0001
            m = rise / run
            b = y0 - m * x0

            lines.append([m, b])
        
        for line in lines:
            self._remove_line(line)


    def _remove_line(self, line):
        m, b = line

        # Place pixels in a cluster where each represents a veritcal line
        all_characters = [pixel for character in self.characters for pixel in character]
        all_characters.sort()

        vertical_lines = []
        previous_x = None
        previous_y = None
        vertical_line = None
        for x, y in all_characters:
            if not vertical_line:
                previous_x = x
                previous_y = y
                vertical_line = [(x, y)]
            elif x != previous_x:
                previous_x = x
                previous_y = y
                vertical_lines.append(vertical_line)
                vertical_line = [(x, y)]
            elif y != previous_y + 1:
                previous_y = y
                vertical_lines.append(vertical_line)
                vertical_line = [(x, y)]
            else:
                previous_y = y
                vertical_line.append((x, y))

        to_remove = []
        for vertical_line in vertical_lines:
            should_remove = True
            for x, y in vertical_line:
                distance = abs(y - m * x - b) / math.sqrt(m**2 + 1)
                if distance >= self.DISTANCE_TO_LINE_THRESHOLD:
                    should_remove = False
                    break
            if should_remove:
                to_remove += vertical_line

        previous = self.characters[:]

        for i, character in enumerate(self.characters[:]):
            new_character = []
            for pixel in character:
                if pixel not in to_remove:
                    new_character.append(pixel)
            self.characters[i] = new_character



    def show_image(self, filename='segmentation.png'):
        """ Displays the resulting segmented image, and saves the file. """
        height, width = self.img.shape

        # White out all of the pixels
        for x in range(width):
            for y in range(height):
                self.img[y, x] = self.WHITE_PIXEL

        for i, character in enumerate(self.characters):
            color = 100 if i % 2 == 0 else 0
            for x, y in character:
                self.img[y, x] = color

        cv2.imshow('segmented', self.img)
        k = cv2.waitKey(0)
        cv2.imwrite('segmented.jpeg', self.img)
