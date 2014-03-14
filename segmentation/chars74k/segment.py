import classify
from collections import defaultdict
import cv2
import fnmatch
import itertools
import math
import numpy as np
import os
from PIL import Image, ImageDraw
import pdb
import sys
import random



class Segmenter:
    """ Given a file, outputs the segmented characters for it. """

    def __init__(self, image, classifier):
        self.img = image

        self.WHITE_PIXEL = 255

        self.WHITE_THRESHOLD = 220
        self.NUM_CHAR_THRESHOLD = 40
        self.NUM_POINTS_FOR_LINE_THRESHOLD = 20
        self.DISTANCE_TO_LINE_THRESHOLD = 3.0

        self.MAX_WIDTH = 400
        self.MAX_HEIGHT = 100

        self.classifier = classifier

        sys.setrecursionlimit(2000)


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

        # If character has multiple separate parts, leave only largest
        self._clean_characters()
        self._remove_noise()
        self._split_connected_characters()
        self._remove_noise()
        pdb.set_trace()
        self.characters.sort()

        return self._create_images()


    def _split_connected_characters(self):
        widths = []
        total_width = 0
        for character in self.characters:
            min_x = min(character, key=lambda c: c[0])[0]
            max_x = max(character, key=lambda c: c[0])[0]
            width = max_x - min_x
            total_width += width
            widths.append(width)
        average_width = total_width / len(self.characters)
        widths_std = np.std(widths)
        lower_width_threshold = average_width - .8 * widths_std
        print lower_width_threshold

        for character in self.characters[:]:
            min_x = min(character, key=lambda c: c[0])[0]
            max_x = max(character, key=lambda c: c[0])[0]
            min_y = min(character, key=lambda c: c[1])[1]
            max_y = max(character, key=lambda c: c[1])[1]

            new_image = np.copy(self.img)
            height, width = new_image.shape
            for x in xrange(width):
                for y in xrange(height):
                    if (x, y) not in character:
                        new_image[y, x] = self.WHITE_PIXEL


            # Get bounding box of the character
            sub_image = np.copy(new_image[min_y:max_y, min_x:max_x])
            sub_image_features = classify.compute_features(sub_image)

            probabilities = self.classifier.predict_proba(sub_image_features)

            if len(probabilities.nonzero()) > 1 and max_x - min_x > average_width:
                new_characters = []
                self._split_character(new_image, min_y, max_y, min_x, max_x, new_characters, character, lower_width_threshold)
                if len(new_characters) > 0:
                    self.characters.remove(character)
                    for new_character in new_characters:
                        self.characters.append(new_character)


    def _split_character(self, image, min_y, max_y, min_x, max_x, new_characters, character, lower_width_threshold):
        if min_x >= max_x: return

        for x_end in range(min_x + 1, max_x + 1):
            if x_end - min_x < lower_width_threshold: continue
            sub_image = np.copy(image[min_y:max_y, min_x:x_end])
            if sub_image.shape[0] == 0 or sub_image.shape[1] == 0:
                pdb.set_trace()
            sub_image_features = classify.compute_features(sub_image)
            probabilities = self.classifier.predict_proba(sub_image_features)[0]
            num_non_zero = len([a for a in probabilities if a != 0])

            if num_non_zero == 1:
                sub_characters = [c for c in character if c[0] <= x_end and c[0] >= min_x]
                new_characters.append(sub_characters)
                self._split_character(image, min_y, max_y, x_end + 1, max_x, new_characters, character, lower_width_threshold)
                return


        # pdb.set_trace()
        # widths_std = np.std(widths)
        # if widths_std > 10:
        #     for i, character in enumerate(self.characters[:]):
        #         if widths[i] > average_width + widths_std * 0.8:
        #             self.characters.remove(character)
        #             sorted_character = sorted(character)
        #             self.characters.append(sorted_character[0:len(sorted_character) / 2])
        #             self.characters.append(sorted_character[len(sorted_character) / 2:])


    def _clean_characters(self):
        # return  # TODO: Remove
        height, width = self.img.shape
        for i, character in enumerate(self.characters[:]):
            character_set = set(character)
            max_group = []
            while len(character_set) > 0:
                group = []
                start_pixel = next(iter(character_set))
                self._create_cluster(start_pixel[0], start_pixel[1], group, character_set, width, height)
                if len(group) > len(max_group):
                    max_group = group
                character_set -= set(group)
            self.characters[i] = max_group



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
        for i in xrange(3):
            lines = cv2.HoughLines(edges, 1, np.pi / 180, (6 - i) * 50)
            if lines != None:
                break
        if lines == None:
            return

        polar_lines = [l[0] for l in lines]
        if polar_lines == None:
            polar_lines = [];
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



    def _create_images(self):
        images = []
        for character in self.characters:
            min_x = min(character, key=lambda c: c[0])[0]
            max_x = max(character, key=lambda c: c[0])[0]
            min_y = min(character, key=lambda c: c[1])[1]
            max_y = max(character, key=lambda c: c[1])[1]

            new_image = np.copy(self.img)
            height, width = new_image.shape
            for x in xrange(width):
                for y in xrange(height):
                    if (x, y) not in character:
                        new_image[y, x] = self.WHITE_PIXEL


            # Get bounding box of the character
            sub_image = np.copy(new_image[min_y:max_y, min_x:max_x])

            images.append(sub_image)
            cv2.imshow('a', sub_image)
            cv2.waitKey(0)

        return images


    def show_image(self, filename='segmentation.png'):
        """ Displays the resulting segmented image, and saves the file. """
        height, width = self.img.shape

        colored_image = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        # White out all of the pixels
        for x in range(width):
            for y in range(height):
                colored_image[y, x] = self.WHITE_PIXEL

        for i, character in enumerate(self.characters):
            color = (255, 166, 0) if i % 2 == 0 else (0, 72, 255)
            for x, y in character:
                colored_image[y, x] = color
        
        cv2.imshow('segmented', colored_image)
        k = cv2.waitKey(0)
        cv2.imwrite('segmented.jpeg', colored_image)
