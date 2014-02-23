clear all;
close all;

% read all images
image_paths = {'small-images/1.jpg', 'small-images/2.jpg'};
images = {};

for i = 1 : length(image_paths)
  images{i} = rgb2gray(im2single(imread(image_paths{i})));
end

image1 = images{1};
image2 = images{2};
image_size = size(image1);

% compute SIFT keypoints
[features1, descriptor1] = vl_sift(image1);
[features2, descriptor2] = vl_sift(image2);
[matches, scores] = vl_ubcmatch(descriptor1, descriptor2);

% filter to best keypoints
best_matches = matches(:, scores < 3000);
best_features1 = features1(:, best_matches(1, :));
best_features2 = features2(:, best_matches(2, :));
num_features = size(best_matches, 2);

% convert keypoints to homogeneous coordinates
best_features1 = [best_features1(1, :); best_features1(2, :); ones(1, num_features)];
best_features2 = [best_features2(1, :); best_features2(2, :); ones(1, num_features)];

% find homography H such that H * best_features1 = best_features2 using RANSAC
threshold = 10;
best_num_inliers = 0;
best_H = [];

for i = 1 : 100
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

  if num_inliers > best_num_inliers
    best_H = H;
    best_num_inliers = num_inliers
  end
end

% scale up by 2 for super resolution image
super_scale = 2;
super_image_size = image_size * super_scale;
super_image = single(zeros(super_image_size));

% proportional gain to multiply error by (similar to PID controller)
gain = 0.8;
iterations = 5;

for i = 1 : iterations
  downsampled_image = super_image(1 : super_scale : end, 1 : super_scale : end);

  % on each iteration, update super_image(j, k) based on the error of sampled images
  for j = 1 : super_image_size(1)
    for k = 1 : super_image_size(2)
      % zero-indexed coordinates for downsampling
      zero_j = j - 1;
      zero_k = k - 1;

      downsampled_point = [floor(zero_k / super_scale) + 1; floor(zero_j / super_scale) + 1; 1];

      % super resolution image is relative to image 1
      image1_point = downsampled_point;
      image2_point = round(best_H * downsampled_point);

      % update super image based on error from image 1
      if image1_point(1) > 0 && image1_point(1) <= size(image1, 2) && image1_point(2) > 0 && image1_point(2) <= size(image1, 1)
        estimated_intensity = downsampled_image(downsampled_point(2), downsampled_point(1));
        actual_intensity = image1(image1_point(2), image1_point(1));
        super_image(j, k) = super_image(j, k) + (actual_intensity - estimated_intensity) * gain;
      end

      % update super image based on error from image 2
      if image2_point(1) > 0 && image2_point(1) <= size(image2, 2) && image2_point(2) > 0 && image2_point(2) <= size(image2, 1)
        estimated_intensity = downsampled_image(downsampled_point(2), downsampled_point(1));
        actual_intensity = image2(image2_point(2), image2_point(1));
        super_image(j, k) = super_image(j, k) + (actual_intensity - estimated_intensity) * gain;
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
