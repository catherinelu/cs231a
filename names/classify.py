from collections import defaultdict
import cv2
import fnmatch
import numpy as np
import itertools
import os
import pdb
import pickle
from scipy import sparse
from sklearn import svm, metrics
import utils

DELETION_COST = 1
INSERTION_COST = 1
CHANGE_COST = 1
NUM_DESCRIPTORS_PER_IMAGE = 2

class Classifier:

    def __init__(self, lexicon):
        """
        Required to pass in a lexicon, which is a list of names corresponding
        to the roster.
        """
        expanded_lexicon = []
        for name in lexicon:
            cur_lexicon = []
            # Enter full name, including middle name
            cur_lexicon.append(name)
            # Enter first name and last name only
            names = name.split()
            if len(names) > 2:
                cur_lexicon.append('%s %s' % (names[0], names[-1]))
            expanded_lexicon.append(cur_lexicon)

        # List of lists, where inner list may contain multiple ways to write the name
        self.lexicon = expanded_lexicon

        self.training_labels = []
        self.training_images = []
        self.PICKLED_IMAGES_FILE = 'training_images'
        self.PICKLED_LABELS_FILE = 'training_labels'
        self.PICKLED_SVM_FILE = 'svm'


    def train(self, directory):
        """
        Goes through all of the .dat files in the directory, and gets those that
        correspond to lowercase/uppercase letters.
        """
        comment_string = '.COMMENT Prompt string: "'
        comment_string_len = len(comment_string)

        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, '*.dat'):
                f = open(os.path.join(root, filename))
                for line in f:
                    if comment_string in line:
                        if line[comment_string_len + 1] == '"' and line[comment_string_len].isalpha():
                            inkml = self._parse_inkml(f)
                            pixels = utils.inkml_to_pixels(inkml)
                            # Change the pixel 2D array into a 1D list
                            chain = list(itertools.chain(*pixels))
                            self.training_labels.append(ord(line[comment_string_len]))
                            self.training_images.append(chain)
                        else:
                            break
                f.close()


        self._add_feature_descriptors()

        # Get training images in correct format
        self.training_images = sparse.csr_matrix(np.array(self.training_images))


    def _add_feature_descriptors(self):
        self.detector = cv2.FeatureDetector_create('ORB')
        self.descriptor = cv2.DescriptorExtractor_create('SIFT')
        self.matcher = cv2.DescriptorMatcher_create('FlannBased')

        # Get two of each label to build the codewords dictionary
        selected_images = defaultdict(list)
        for i, label in enumerate(self.training_labels):
            if len(selected_images[label]) < 2:
                selected_images[label].append(i)

        # For each pair of the selected images, get their descriptors and
        # use the top N descriptors
        self.codewords = []
        for label in selected_images:
            for image1, image2 in selected_images[label]:
                image1_descriptor = self._get_descriptor(image1)
                image2_descriptor = self._get_descriptor(image2)


                matches = self.matcher.match(image1_descriptor, image2_descriptor)
                top_matches = sorted(matches, key=lambda match: match.distance)[0:NUM_DESCRIPTORS_PER_IMAGE]
                self.codewords += top_matches

        pdb.set_trace()


    def _get_descriptor(image):
        pixels = np.resize(image, (utils.SIDE_LEN, utils.SIDE_LEN))
        keypoints = self.detector.detect(pixels)
        page_kp, page_descriptor = self.descriptor.compute(pixels, keypoints)


    def load_training_data(self):
        """ Loads training data from pickled files. """
        d = open(self.PICKLED_IMAGES_FILE)
        self.training_images = pickle.load(d)
        d.close()

        l = open(self.PICKLED_LABELS_FILE)
        self.training_labels = pickle.load(l)
        l.close()


    def load_svm(self):
        s = open(self.PICKLED_SVM_FILE)
        self.svm = pickle.load(s)
        s.close()


    def train_svm(self):
        # Train based on the training labels and training images
        self.svm = svm.SVC(gamma=0.001, probability=True)
        self.svm.fit(self.training_images, self.training_labels)


    def pickle_training_data(self):
        """ Pickle training data. Must run read_training_data_from_dataset first. """
        d = open(self.PICKLED_IMAGES_FILE, 'w')
        pickle.dump(self.training_images, d)
        d.close()

        l = open(self.PICKLED_LABELS_FILE, 'w')
        pickle.dump(self.training_labels, l)
        l.close()


    def pickle_svm(self):
        s = open(self.PICKLED_SVM_FILE, 'w')
        pickle.dump(self.svm, s)
        s.close()


    def classify(self, characters):
        characters = [utils.normalize(c) for c in characters]

        characters = [list(itertools.chain(*c)) for c in characters]
        characters = sparse.csr_matrix(np.array(characters))
        predicted = self.svm.predict(characters)
        predicted_name = ''.join([chr(p) for p in predicted])
        print 'Predicted:', predicted_name

        roster_name = self._get_closest_name(predicted_name)
        return roster_name

        # print self.svm.score(characters, expected)
        # print('Classification report for classifier %s:\n%s\n'
        #   % (self.svm, metrics.classification_report(expected, predicted)))


    def _get_closest_name(self, predicted_name):
        shortest_edit_distance = float('Inf')
        roster_name = None

        for i, names in enumerate(self.lexicon):
            for name in names:
                name = name.replace(' ', '')
                cur_edit_distance = self._recursive_shortest_edit_distance(predicted_name,
                    len(predicted_name) - 1, name, len(name) - 1, 0, 0, 0, shortest_edit_distance)
                if cur_edit_distance < shortest_edit_distance:
                    shortest_edit_distance = cur_edit_distance
                    roster_name = self.lexicon[i][0]

        print shortest_edit_distance, roster_name
        return roster_name

    
    def _recursive_shortest_edit_distance(self, predicted_name, predicted_name_length,
        possible_name, possible_name_length, i, j, cur_edit_distance, shortest_edit_distance):
        if i == predicted_name_length and j == possible_name_length:
            return cur_edit_distance
        elif i == predicted_name_length:
            return cur_edit_distance + (possible_name_length - j) * INSERTION_COST
        elif j == possible_name_length:
            return cur_edit_distance + (predicted_name_length - i) * DELETION_COST
        elif cur_edit_distance > shortest_edit_distance:  # Short-circuit
            return cur_edit_distance

        if predicted_name[i].lower() == possible_name[j].lower():
            return self._recursive_shortest_edit_distance(predicted_name, predicted_name_length,
                possible_name, possible_name_length, i + 1, j + 1, cur_edit_distance, shortest_edit_distance)
        else:
            delete_edit_distance = self._recursive_shortest_edit_distance(predicted_name,
                predicted_name_length, possible_name, possible_name_length, i + 1, j,
                cur_edit_distance + DELETION_COST, shortest_edit_distance)
            insertion_cost = self._recursive_shortest_edit_distance(predicted_name,
                predicted_name_length, possible_name, possible_name_length, i, j + 1,
                cur_edit_distance + INSERTION_COST, shortest_edit_distance)
            change_edit_distance = self._recursive_shortest_edit_distance(predicted_name,
                predicted_name_length, possible_name, possible_name_length, i + 1, j + 1,
                cur_edit_distance + CHANGE_COST, shortest_edit_distance)
            return min(delete_edit_distance, insertion_cost, change_edit_distance)


    def _parse_inkml(self, open_file):
        inkml_character = []

        in_stroke = False
        inkml_stroke = []
        for line in open_file:
            if '.PEN_DOWN' in line:
                in_stroke = True
            elif '.PEN_UP' in line:
                in_stroke = False
                inkml_character.append(inkml_stroke)
                inkml_stroke = []
            elif in_stroke:
                coordinates = line.split()
                inkml_stroke.append((int(coordinates[0]), int(coordinates[1])))
        return inkml_character
