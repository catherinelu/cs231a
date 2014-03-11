from collections import defaultdict
import cv2
import fnmatch
import numpy as np
import itertools
import os
import pdb
import pickle
from scipy import sparse
from sklearn import svm, metrics, cluster
from skimage.feature import hog
from skimage import exposure
import utils


import pylab as pl

from sklearn.svm import SVC
from sklearn.preprocessing import Scaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt


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
        self.large_training_images = []  # Used for finding SIFT descriptors
        self.PICKLED_IMAGES_FILE = 'training_images'
        self.PICKLED_LABELS_FILE = 'training_labels'
        self.PICKLED_SVM_FILE = 'svm'

        self.detector = cv2.FeatureDetector_create('SIFT')
        self.descriptor = cv2.DescriptorExtractor_create('SIFT')
        self.matcher = cv2.DescriptorMatcher_create('FlannBased')

        self.LARGE_IMAGE_SIZE = 64


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
                            # TODO: Comment back in if you want to use the pixel features
                            # pixels = utils.inkml_to_pixels(inkml)
                            # # Change the pixel 2D array into a 1D list
                            # chain = list(itertools.chain(*pixels))
                            self.training_labels.append(ord(line[comment_string_len]))
                            # self.training_images.append(chain)

                            self.training_images.append([])

                            original_side_len = utils.SIDE_LEN
                            utils.SIDE_LEN = self.LARGE_IMAGE_SIZE
                            large_pixels = utils.inkml_to_pixels(inkml)
                            chain = list(itertools.chain(*large_pixels))
                            self.large_training_images.append(chain)
                            utils.SIDE_LEN = original_side_len
                        else:
                            break
                f.close()


        # KMeans:
        # self._create_codewords_with_kmeans()

        # for i, image in enumerate(self.large_training_images[:]):
        #     self.training_images[i] += self._add_feature_descriptors(image)


        # HOG:
        for i, image in enumerate(self.large_training_images[:]):
            hog_features = self._add_hog_features(image)
            # pdb.set_trace()
            self.training_images[i] = hog_features.tolist()



        # Get training images in correct format
        # self.training_images = sparse.csr_matrix(np.array(self.training_images, dtype=np.uint8))


    def _create_codewords_with_kmeans(self):
        all_descriptors = []
        for image in self.large_training_images:
            keypoints, descriptors = self._get_descriptor(image)
            if descriptors != None:
                for descriptor in descriptors:
                    all_descriptors.append(descriptor)

        kmeans = cluster.KMeans(n_clusters=52, init='k-means++', max_iter=100, n_init=1)
        kmeans.fit(all_descriptors)

        self.codewords  = kmeans.cluster_centers_


    def _create_codewords(self):
        # Get two of each label to build the codewords dictionary
        selected_images = defaultdict(list)
        for i, label in enumerate(self.training_labels):
            if len(selected_images[label]) < 2:
                selected_images[label].append(i)

        # For each pair of the selected images, get their descriptors and
        # use the top N descriptors
        self.codewords = []
        for label in selected_images:
            image1, image2 = self.large_training_images[selected_images[label][0]], self.large_training_images[selected_images[label][1]]
            image1_keypoints, image1_descriptors = self._get_descriptor(image1)
            image2_keypoints, image2_descriptors = self._get_descriptor(image2)

            if image1_descriptors == None and image2_descriptors == None:
                print 'NO DESCRIPTORS FOUND'
                continue
            elif image1_descriptors == None:
                self.codewords.append(image2_descriptors[0:NUM_DESCRIPTORS_PER_IMAGE])
                print 'IMAGE1 DESCRIPTORS NOT FOUND'
                continue
            elif image2_descriptors == None:
                self.codewords.append(image1_descriptors[0:NUM_DESCRIPTORS_PER_IMAGE])
                print 'IMAGE2 DESCRIPTORS NOT FOUND'
                continue

            matches = self.matcher.match(image1_descriptors, image2_descriptors)
            top_matches = sorted(matches, key=lambda match: match.distance)[0:NUM_DESCRIPTORS_PER_IMAGE]
            descriptors = [image1_descriptors[m.queryIdx] for m in top_matches]
            self.codewords += descriptors


    def _get_descriptor(self, image):
        pixels = np.array(utils.change_pixel_values(image), dtype=np.uint8)
        pixels.resize((utils.SIDE_LEN, utils.SIDE_LEN))
        keypoints = self.detector.detect(pixels)
        return self.descriptor.compute(pixels, keypoints)


    def _add_hog_features(self, image):
        # pdb.set_trace()
        normalized_image = np.resize(np.array(image), (self.LARGE_IMAGE_SIZE, self.LARGE_IMAGE_SIZE))
        hog_image = hog(normalized_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2))


        # To view the histogram, you can uncomment out this
        # _, hog_image = hog(normalized_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualise=True)
        # plt.figure(figsize=(8, 4))

        # plt.subplot(121).set_axis_off()
        # pdb.set_trace()
        # plt.imshow(normalized_image, cmap=plt.cm.gray)
        # plt.title('Input image')

        # # Rescale histogram for better display
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

        # plt.subplot(122).set_axis_off()
        # plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        # plt.title('Histogram of Oriented Gradients')
        # plt.show()

        # hog_image.resize(hog_image.size)

        return hog_image


    def _add_feature_descriptors(self, image):
        feature_descriptors = [-1 for i in xrange(len(self.codewords))]
        
        keypoints, descriptors = self._get_descriptor(image)

        if descriptors == None:
            return feature_descriptors

        for descriptor in descriptors:
            for i, codeword in enumerate(self.codewords):
                distance = [np.linalg.norm(descriptor - codeword) for codeword in self.codewords]
                if distance == -1 or distance < feature_descriptors[i]:
                    feature_descriptors[i] = distance

        return feature_descriptors


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
        self.svm = svm.SVC(gamma=0.001)  # TODO: Re-add probability=True
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


    def tune_svm(self):
        pdb.set_trace()
        C_range = 10. ** np.arange(-2, 9)
        gamma_range = 10. ** np.arange(-5, 4)

        X = self.training_images
        Y = self.training_labels

        param_grid = dict(gamma=gamma_range, C=C_range)

        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(y=Y, n_folds=5))

        grid.fit(X, Y)

        print("The best classifier is: ", grid.best_estimator_)

        # plot the scores of the grid
        # grid_scores_ contains parameter settings and scores
        score_dict = grid.grid_scores_

        # We extract just the scores
        scores = [x[1] for x in score_dict]
        scores = np.array(scores).reshape(len(C_range), len(gamma_range))

        # Make a nice figure
        pl.figure(figsize=(8, 6))
        pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
        pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
        pl.xlabel('gamma')
        pl.ylabel('C')
        pl.colorbar()
        pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        pl.yticks(np.arange(len(C_range)), C_range)
        pl.show()


    def cross_validate(self, n=10):
        num_training_images = len(self.training_images)
        n_training_image_groups = [None for i in xrange(n)]
        n_training_label_groups = [None for i in xrange(n)]

        for i in xrange(n):
            start_index = int(float(i) / n * num_training_images)
            end_index = int(float(i + 1) / n * num_training_images)
            n_training_image_groups[i] = self.training_images[start_index:end_index]
            n_training_label_groups[i] = self.training_labels[start_index:end_index]


        num_total_correct = 0
        num_total_incorrect = 0
        for i in xrange(n):
            print 'Testing fold %d...' % i

            num_correct = 0
            num_incorrect = 0

            svm_i = SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                        gamma=0.001, kernel='rbf', max_iter=-1, probability=False,
                        random_state=None, shrinking=True, tol=0.001, verbose=False)

            train_images = []
            train_labels = []
            for j in xrange(n):
                if j != i:
                    train_images += n_training_image_groups[j]
                    train_labels += n_training_label_groups[j]
            svm_i.fit(sparse.csr_matrix(np.array(train_images, dtype=np.uint8)), train_labels)

            test_images = sparse.csr_matrix(np.array(n_training_image_groups[i], dtype=np.uint8))
            test_labels = n_training_label_groups[i]
            predicted_labels = svm_i.predict(test_images)

            for i, predicted_label in enumerate(predicted_labels):
                if predicted_label == test_labels[i]:
                    num_correct += 1
                else:
                    num_incorrect += 1

            print '%d correct and %d incorrect (%d%% accuracy)' % (num_correct,
                num_incorrect, int(num_correct / float(num_correct + num_incorrect) * 100))

            num_total_correct += num_correct
            num_total_incorrect += num_incorrect

        print 'Final stats:'
        print '%d total correct and %d total incorrect (%d%% accuracy)' % (num_total_correct,
            num_total_incorrect, int(num_total_correct / float(num_total_correct + num_total_incorrect) * 100))



    def classify(self, characters):
        characters = [utils.normalize(c) for c in characters]
        for character in characters:
            character += self._add_feature_descriptors(character)
        characters = [list(itertools.chain(*c)) for c in characters]
        characters = sparse.csr_matrix(np.array(characters, dtype=np.uint8))
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
