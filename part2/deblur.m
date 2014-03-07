clear all;
close all;

image = im2single(imread('blurred.jpg'));
image_size = size(image);
deblurred_image = single(zeros(image_size));

% scale up by 2 for super resolution image
% super_scale = 2;
% super_size = image_size * super_scale;
% super_image = single(zeros(super_size));

% proportional gain to multiply error by (similar to PID controller)
c = 10;
iterations = 50;

point_spread_filter = 1 / 15 * [1 2 1; 2 3 2; 1 2 1];
averaging_filter = 1 / 5 * [0 1 0; 1 1 1; 0 1 0];

for i = 1 : iterations
  deblurred_image_after_psf = conv2(deblurred_image, point_spread_filter, 'same');
  deblurred_image = deblurred_image + conv2(image - deblurred_image_after_psf, point_spread_filter / c, 'same');

  % super_image_after_psf = conv2(super_image, point_spread_filter, 'same');
  % averaged_super_image = conv2(super_image_after_psf, averaging_filter, 'same');
  % downsampled_image = averaged_super_image(1 : super_scale : end, 1 : super_scale : end);

% 
%   % on each iteration, update super_image(j, k) based on the error of sampled images
%   for j = 1 : super_size(1)
%     for k = 1 : super_size(2)
%       % zero-indexed coordinates for downsampling
%       zero_j = j - 1;
%       zero_k = k - 1;
% 
%       downsampled_center = [floor(zero_k / super_scale) + 1; floor(zero_j / super_scale) + 1; 1];
%       downsampled_points = [];
%       valid_points = [];
% 
%       for l = -1 : 1
%         for m = -1 : 1
%           new_point = downsampled_center + [m; l; 0];
%           downsampled_points = [downsampled_points, new_point];
%           if new_point(1) > 0 && new_point(1) <= image_size(2) && new_point(2) > 0 && new_point(2) <= image_size(1)
%             valid_points = [valid_points true];
%           else
%             valid_points = [valid_points false];
%           end
%         end
%       end
% 
%       valid_points = valid_points & 1;
%       downsampled_points = downsampled_points(:, valid_points);
%       num_downsampled_points = size(downsampled_points, 2);
% 
%       weights = weights(valid_points);
%       sum_root_weights = sqrt(sum(weights));
% 
%       for m = 1 : num_downsampled_points
%         % compute point corresponding to downsampled_point in current image
%         image_point = round(transformations{l} * downsampled_points(:, m));
% 
%         % update super image based on error from the current image
%         if image_point(1) > 0 && image_point(1) <= size(images{l}, 2) && image_point(2) > 0 && image_point(2) <= size(images{l}, 1)
%           estimated_intensity = downsampled_image(downsampled_points(2, m), downsampled_points(1, m));
%           actual_intensity = images{l}(image_point(2), image_point(1));
%           super_image(j, k) = super_image(j, k) + (actual_intensity - estimated_intensity) * weights(m) / (c * sum_root_weights);
%         end
%       end
%     end
%   end

  % figure;
  % imshow(deblurred_image);

  i
end

figure;
imshow(image);
title('Image');

figure;
imshow(deblurred_image);
title('Deblurred');
imwrite(deblurred_image, 'deblurred.jpg');
