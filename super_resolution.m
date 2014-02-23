clear all;
close all;

% read all images
image_paths = {'smaller-images/1.jpg', 'smaller-images/2.jpg', 'smaller-images/3.jpg'};
num_images = length(image_paths);

images = {};
for i = 1 : num_images
  images{i} = rgb2gray(im2single(imread(image_paths{i})));
end

image1 = images{1};
image_size = size(image1);

transformations = {};
transformations{1} = eye(3);
[features1, descriptor1] = vl_sift(image1);

% find transformation from image1 to all other images
for i = 2 : num_images
  % compute SIFT keypoints
  [features2, descriptor2] = vl_sift(images{i});
  [matches, scores] = vl_ubcmatch(descriptor1, descriptor2);

  % filter to best keypoints
  best_matches = matches;
  best_features1 = features1(:, best_matches(1, :));
  best_features2 = features2(:, best_matches(2, :));
  num_features = size(best_matches, 2)

  % convert keypoints to homogeneous coordinates
  best_features1 = [best_features1(1, :); best_features1(2, :); ones(1, num_features)];
  best_features2 = [best_features2(1, :); best_features2(2, :); ones(1, num_features)];

  % find homography H such that H * best_features1 = best_features2 using RANSAC
  threshold = 10;
  best_num_inliers = 0;
  best_total_score = inf;

  for j = 1 : 1000
    subset = vl_colsubset(1 : num_features, 4);
    best_features1_subset = best_features1(:, subset);
    best_features2_subset = best_features2(:, subset);

    % We can convert this to best_features1' * H' = best_features2'. Hence,
    % H = (best_features1' \ best_features2')'.
    H = (best_features1_subset' \ best_features2_subset')';

    % determine score and inliers
    score = (H * best_features1 - best_features2) .^ 2;
    score = sqrt(sum(score, 1));

    inliers = score <= threshold;
    num_inliers = sum(inliers);
    total_score = sum(score(inliers));

    % TODO: improve this: for the same number of inliers, take transform w/ lowest error
    if num_inliers > best_num_inliers || (num_inliers == best_num_inliers && total_score < best_total_score)
      best_H = H;
      best_total_score = total_score;
      best_num_inliers = num_inliers
    end
  end

  transformations{i} = best_H / best_H(3, 3);
  % transformed_image = zeros(image_size);

  % for k = 1 : image_size(1)
  %   for l = 1 : image_size(2)
  %     point = [l; k; 1];
  %     transformed_point = round(transformations{i} * point);

  %     if transformed_point(1) > 0 && transformed_point(1) <= image_size(2) && transformed_point(2) > 0 && transformed_point(2) <= image_size(1)
  %       transformed_image(transformed_point(2), transformed_point(1)) = image1(point(2), point(1));
  %     end
  %   end
  % end

  % figure;
  % imshow(transformed_image);

  % figure;
  % imshow(images{i});
end

% scale up by 2 for super resolution image
super_scale = 2;
super_size = image_size * super_scale;
super_image = single(zeros(super_size));

% proportional gain to multiply error by (similar to PID controller)
c = 5;
iterations = 5;

psf = 1 / 15 * [1 2 1; 2 3 2; 1 2 1];

for i = 1 : iterations
  super_image_after_psf = conv2(super_image, psf, 'same');
  downsampled_image = super_image_after_psf(1 : super_scale : end, 1 : super_scale : end);

  % on each iteration, update super_image(j, k) based on the error of sampled images
  for j = 1 : super_size(1)
    for k = 1 : super_size(2)
      % zero-indexed coordinates for downsampling
      zero_j = j - 1;
      zero_k = k - 1;

      downsampled_center = [floor(zero_k / super_scale) + 1; floor(zero_j / super_scale) + 1; 1];
      downsampled_points = [];
      valid_points = [];
      weights = reshape(psf, 1, 9);

      for l = -1 : 1
        for m = -1 : 1
          new_point = downsampled_center + [m; l; 0];
          downsampled_points = [downsampled_points, new_point];
          if new_point(1) > 0 && new_point(1) <= image_size(2) && new_point(2) > 0 && new_point(2) <= image_size(1)
            valid_points = [valid_points true];
          else
            valid_points = [valid_points false];
          end
        end
      end

      valid_points = valid_points & 1;
      downsampled_points = downsampled_points(:, valid_points);
      num_downsampled_points = size(downsampled_points, 2);

      weights = weights(valid_points);
      sum_root_weights = sqrt(sum(weights));

      % num_downsampled_points = 1;
      % downsampled_points = [downsampled_center];
      % weights = [1];
      % sum_weights = sum(weights);

      for l = 1 : num_images
        for m = 1 : num_downsampled_points
          % compute point corresponding to downsampled_point in current image
          image_point = round(transformations{l} * downsampled_points(:, m));

          % update super image based on error from the current image
          if image_point(1) > 0 && image_point(1) <= size(images{l}, 2) && image_point(2) > 0 && image_point(2) <= size(images{l}, 1)
            estimated_intensity = downsampled_image(downsampled_points(2, m), downsampled_points(1, m));
            actual_intensity = images{l}(image_point(2), image_point(1));
            super_image(j, k) = super_image(j, k) + (actual_intensity - estimated_intensity) * weights(m) / (c * sum_root_weights);
          end
        end
      end
    end
  end

  figure;
  imshow(super_image);

  i
end

figure;
imshow(super_image);
imwrite(super_image, 'super.jpg');
