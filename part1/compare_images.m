close all;
clear all;

% where page JPEGs are located
page_dir = '/Users/karthikv/Dropbox/Shared/CS231A/scoryst-project/cs221-exams-better';
prefixes = {'adam', 'ahmed', 'alp', 'andy', 'arushi', 'brett'};
num_prefixes = length(prefixes);

% page suffixes
pages = [0 : 5];
num_pages = length(pages);

images = cell(num_prefixes, num_pages);
image_features = cell(num_prefixes, num_pages);
image_descriptors = cell(num_prefixes, num_pages);

% read real exam images, compute SIFT features/descriptors
for i = 1 : num_prefixes
  for j = 1 : num_pages
    path = [page_dir '/' prefixes{i} int2str(pages(j)) '.jpg'];
    images{i}{j} = im2single(imread(path));

    % convert to grayscale if necessary
    if size(images{i}{j}, 3) == 3
      images{i}{j} = rgb2gray(images{i}{j});
    end

    % resize image to width of 500
    image_size = size(images{i}{j});
    images{i}{j} = imresize(images{i}{j}, 500 / image_size(2));

    [features, descriptor] = vl_sift(images{i}{j});
    image_features{i}{j} = features;
    image_descriptors{i}{j} = descriptor;

    fprintf('Finished computing SIFT descriptors for prefix %s, page %g\n', prefixes{i}, pages(j));
  end
end

blank_suffix = 'blank';
blank_images = cell(num_pages);
blank_image_features = cell(num_pages);
blank_image_descriptors = cell(num_pages);

% read blank exam images, compute SIFT keypoints
for i = 1 : num_pages
  path = [page_dir '/' int2str(pages(i)) blank_suffix '.jpg'];
  blank_images{i} = im2single(imread(path));

  % convert to grayscale if necessary
  if size(blank_images{i}, 3) == 3
    blank_images{i} = rgb2gray(blank_images{i});
  end

  % resize image to width of 500
  image_size = size(blank_images{i});
  blank_images{i} = imresize(blank_images{i}, 500 / image_size(2));

  [features, descriptor] = vl_sift(blank_images{i});
  blank_image_features{i} = features;
  blank_image_descriptors{i} = descriptor;

  fprintf('Finished computing SIFT descriptors for blank page %g\n', pages(i));
end

% statistics to report
total_positives = num_prefixes * num_pages;
total_negatives = num_prefixes * num_pages * num_pages - total_positives;
true_positives = 0;
true_negatives = 0;

for i = 1 : num_prefixes
  for j = 1 : num_pages
    for k = 1 : num_pages
      blank_features = blank_image_features{k};
      blank_descriptors = blank_image_descriptors{k};

      features = image_features{i}{j};
      descriptors = image_descriptors{i}{j};

      % match descriptors and filter to best matches
      [matches, scores] = vl_ubcmatch(blank_descriptors, descriptors);
      best_matches = matches(:, scores < 5000);

      best_blank_features = blank_features(:, best_matches(1, :));
      best_features = features(:, best_matches(2, :));
      num_features = size(best_features, 2);

      % convert keypoints to homogeneous coordinates
      best_blank_features = [best_blank_features(1, :); best_blank_features(2, :); ...
        ones(1, num_features)];
      best_features = [best_features(1, :); best_features(2, :); ones(1, num_features)];

      % find homography H such that H * best_blank_features = best_features
      [H, num_inliers, score] = estimate_homography(best_blank_features, best_features, 100);

      % compute scale and rotation from homography
      scale = sqrt(H(1, 1) ^ 2 + H(1, 2) ^ 2);
      rotation = acos(H(1, 1) / scale);

      % debugging information:
      % H
      % scale
      % rotation
      % num_features
      % num_inliers

      % if scale is close to 1 and rotation is close to 0, we've found a valid match
      if abs(scale - 1) < 0.3 && abs(rotation) < 0.3
        if j ~= k
          % failed test if pages are actually different
          result = 'FAIL';
        else
          result = 'pass';
          true_positives = true_positives + 1;
        end

        fprintf('[%s] %s page %g matches blank page %g\n', result, prefixes{i}, ...
          pages(j), pages(k));
      else
        if j == k
          % failed test if pages are actually the same
          result = 'FAIL';
        else
          result = 'pass';
          true_negatives = true_negatives + 1;
        end

        fprintf('[%s] %s page %g does NOT match blank page %g\n', result, prefixes{i}, ...
          pages(j), pages(k));
      end
    end
  end
end

% report statistics
fprintf('\n---\n\n')
fprintf('True positives: %g/%g = %g\n', true_positives, total_positives, ...
  true_positives / total_positives);
fprintf('True negatives: %g/%g = %g\n', true_negatives, total_negatives, ...
  true_negatives / total_negatives);
