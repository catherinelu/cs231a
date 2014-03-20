clear all;
close all;

addpath('../perspective-correction');
addpath('../map-questions');

NUM_FRAMES_TO_SKIP = 5;

video = cv.VideoCapture('./paper.mov');
pages = {};

% SIFT features/descriptors from the previous frame
previous_features = false;
previous_descriptors = false;

% whether the last affine transform was invalid
last_transform_invalid = false;

while video.grab()
  image = video.retrieve();
  image_gray = rgb2gray(im2single(image));

  image_gray = imresize(image_gray, 500 / size(image_gray, 2));
  [features, descriptors] = vl_sift(image_gray);

  if previous_features == false
    % no previous image; this must be the first page
    pages{end + 1} = image;
  else
    % TODO: should decompose this
    [matches, scores] = vl_ubcmatch(previous_descriptors, descriptors);
    best_matches = matches(:, scores < 5000);

    best_previous_features = previous_features(:, best_matches(1, :));
    best_features = features(:, best_matches(2, :));
    num_features = size(best_features, 2);

    % convert to homogeneous coordinates
    best_previous_features = [best_previous_features(1, :); best_previous_features(2, :); ...
      ones(1, num_features)];
    best_features = [best_features(1, :); best_features(2, :); ones(1, num_features)];

    [H, num_inliers, score] = estimate_homography(best_previous_features, best_features, 10000);
    scale = sqrt(H(1, 1) ^ 2 + H(1, 2) ^ 2);
    rotation = acos(H(1, 1) / scale);

    % Look for when we can't make a transformation from the current frame to
    % the previous frame. This implies that a page is being flipped. Take the
    % frame that comes after a streak of invalid transformations, as this will
    % represent the next page when flipping is complete.
    if abs(scale - 1) < 0.3 && abs(rotation) < 0.3 && num_inliers > num_features / 3
      % valid transformation found!
      if last_transform_invalid
        fprintf('last invalid, saving page!\n')
        last_transform_invalid = false;
        pages{end + 1} = image;
        figure, imshow(image);
      end
    else
      fprintf('invalid transform\n')
      H
      num_inliers
      num_features
      last_transform_invalid = true;
    end
  end

  previous_features = features;
  previous_descriptors = descriptors;

  fprintf('processed frame\n')

  % skip NUM_FRAMES_TO_SKIP; we don't need so much information
  for i = 1 : NUM_FRAMES_TO_SKIP - 1
    if ~video.grab()
      break
    end
  end
end

% apply perspective correction to the images
for i = 1 : length(pages)
  corrected_image = correct_perspective(pages{i});
  size(corrected_image)

  % might not be able to find a rectangle in the image 
  if corrected_image ~= false
    figure, imshow(corrected_image);
  end
end

addpath('../map-questions');
rmpath('../perspective-correction');
