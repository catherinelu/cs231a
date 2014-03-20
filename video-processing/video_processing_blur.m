% clear all;
% close all;
% 
% addpath('../perspective-correction');
% addpath('../map-questions');
% 
% VIDEO_PATH = './lecture.mov';
% video = cv.VideoCapture(VIDEO_PATH);
% pages = {};
% 
% % whether the last affine transform was invalid
% % last_transform_invalid = false;
% 
% % number of consecutive frames with a valid transform needed to be deemed stable
% num_frames_for_stability = 5;
% 
% % number of consecutive frames with a valid transform
% num_frames_valid = 0;
% 
% frame = 1;
% num_frames = 0;
% blurs = [0];
% 
% previous_image = false;
% 
% while video.grab()
%   image = video.retrieve();
%   image_gray = cv.cvtColor(image, 'BGR2GRAY');
% 
%   if size(previous_image, 1) > 1
%     M = cv.estimateRigidTransform(previous_image, image);
% 
%     if size(M, 1) == 0
%       blurs = [blurs blurs(frame - 1)];
%     else
%       estimated_page = cv.warpAffine(previous_image, M);
%       estimated_page_gray = cv.cvtColor(estimated_page, 'BGR2GRAY');
% 
%       page_gray = cv.cvtColor(image, 'BGR2GRAY');
%       page_gray(estimated_page_gray == 0) = 0;
% 
%       image_area = size(estimated_page_gray, 1) * size(estimated_page_gray, 2);
%       blurs = [blurs sum(sum(estimated_page_gray - page_gray))];
%     end
%   end
% 
%   previous_image = image;
%   frame = frame + 1;
% 
%   % blur = fft2(image_gray);
%   % blur = sum(sum(abs(blur)));
%   % blurs = [blurs, (sum(sum(cv.Canny(image, [50 200])))) / (size(image, 1) * size(image, 2))];
%   % blurs = [blurs blur];
%   num_frames = num_frames + 1;
% end


normalized_blurs = blurs / max(blurs);
figure, plot(1 : num_frames, normalized_blurs);
grid on;
xlabel('Frame number')
ylabel('Normalized difference')
title('Normalized difference vs. Frame number')

max_blur = max(normalized_blurs);
min_blur = min(normalized_blurs);

target_peaks = [55, 143, 215, 295, 365, 435, 545, 610, 670, 750, 840, 920, 1010, 1115, 1190, 1280];

for i = 1 : length(target_peaks)
  hold on;
  plot([target_peaks(i) target_peaks(i)], [min_blur max_blur], 'r');
end

thresholded = zeros(1, length(normalized_blurs));

frames_to_converge = 15;
num_converging_frames = frames_to_converge;

for i = 1 : length(normalized_blurs)
  if normalized_blurs(i) > 0.5
    num_converging_frames = 0;
    thresholded(i) = 1;
  else
    num_converging_frames = num_converging_frames + 1;
    if num_converging_frames >= frames_to_converge
      thresholded(i) = 0;
    else
      thresholded(i) = 1;
    end
  end
end

frames_to_pick = [1];
for i = 1 : length(thresholded) - 1
  if thresholded(i) == 1 && thresholded(i + 1) == 0
    frames_to_pick = [frames_to_pick (i + 1)];
  end
end

figure, plot(1 : num_frames, thresholded);
grid on;
xlabel('Frame number')
ylabel('Thresholded difference')
title('Thresholded difference vs. Frame number')

for i = 1 : length(target_peaks)
  hold on;
  plot([target_peaks(i) target_peaks(i)], [0 1], 'r');
end

% [sorted_blurs, sort_indexes] = sort(blurs);
% tenth_percentile_index = length(sort_indexes) / 5;
% 
% % reset video
% video = cv.VideoCapture(VIDEO_PATH);
% 
% while video.grab()
%   image = video.retrieve();
%   blur = blurs(frame);
%   blur_index = find(sort_indexes == frame);
% 
%   if blur_index < tenth_percentile_index
%     'blurry!'
%     num_frames_valid = 0;
%   else
%     num_frames_valid = num_frames_valid + 1;
% 
%     if num_frames_valid == num_frames_for_stability
%       'not blurry + stable!'
%       pages{end + 1} = image;
%     end
%   end
% 
%   % M = cv.estimateRigidTransform(image, previous_image);
% 
%   % Look for when we can't make an affine transform from the current frame to
%   % the previous frame. This implies that a page is being flipped (out of
%   % plane transformation is not affine). Take the frame that comes after
%   % a streak of invalid transformations, as this will represent the next page
%   % when flipping is complete.
%   % if size(M, 1) == 0
%   %   'invalid'
%   %   % last_transform_invalid = true;
%   %   num_frames_valid = 0;
%   % else
%   %   % last_transform_invalid = false;
%   %   num_frames_valid = num_frames_valid + 1;
% 
%   %   if num_frames_valid == num_frames_for_stability
%   %     'stable!'
%   %     pages{end + 1} = image;
%   %   end
%   % end
% 
%   frame = frame + 1;
% end
% 
% previous_image = false;
% previous_features = false;
% previous_descriptors = false;
% 
% % apply perspective correction to the images
% for i = 1 : length(pages)
%   corrected_image = correct_perspective(pages{i});
%   duplicate_frame = false;
% 
%   if corrected_image == false
%     continue
%   end
% 
%   if size(previous_image, 1) > 1
%     % 'previous image not false'
%     % [features, descriptors] = vl_sift(rgb2gray(im2single(pages{i})));
% 
%     % % match descriptors and filter to best matches
%     % [matches, scores] = vl_ubcmatch(previous_descriptors, descriptors);
%     % best_matches = matches(:, scores < 5000);
% 
%     % best_previous_features = previous_features(:, best_matches(1, :));
%     % best_features = features(:, best_matches(2, :));
%     % num_features = size(best_features, 2);
% 
%     % % convert keypoints to homogeneous coordinates
%     % best_previous_features = [best_previous_features(1, :); best_previous_features(2, :); ...
%     %   ones(1, num_features)];
%     % best_features = [best_features(1, :); best_features(2, :); ones(1, num_features)];
% 
%     % % find homography H such that H * best_blank_features = best_features
%     % [H, num_inliers, score] = estimate_homography(best_previous_features, best_features, 100);
% 
%     % % compute scale and rotation from homography
%     % scale = sqrt(H(1, 1) ^ 2 + H(1, 2) ^ 2);
%     % rotation = acos(H(1, 1) / scale);
% 
%     % % debugging information:
%     % H
%     % scale
%     % rotation
%     % num_features
%     % num_inliers
% 
%     % % if scale is close to 1 and rotation is close to 0, we've found a valid match;
%     % % add it to the consideration set
%     % if abs(scale - 1) < 0.6 && abs(rotation) < 1.0 % && num_inliers > 0
%     %   duplicate_frame = true
%     % end
%   end
% 
%   if duplicate_frame
%     'found affine transform; skipping'
%   else
%     figure, imshow(corrected_image);
%   end
% 
%   previous_image = pages{i};
%   [previous_features, previous_descriptors] = vl_sift(rgb2gray(im2single(pages{i})));
% end
% 
% % accumulate corrected pages
% % filtered_pages = {};
% % corrected_pages = {};
% % 
% % for i = 1 : length(pages)
% %   page = pages{i};
% %   corrected_page = correct_perspective(page);
% % 
% %   if size(corrected_page, 1) > 1
% %     filtered_pages{end + 1} = page;
% %     corrected_pages{end + 1} = corrected_page;
% %   end
% % end
% % 
% % best_pages = {};
% % prev_page = filtered_pages{1};
% % best_pages{1} = prev_page;
% 
% % eliminate pages with affine transforms between them
% % for i = 2 : length(filtered_pages)
% %   page = filtered_pages{i};
% %   M = cv.estimateRigidTransform(prev_page, page);
% % 
% %   if size(M, 1) == 0
% %     best_pages{end + 1} = page;
% %   else
% %     estimated_page = cv.warpAffine(prev_page, M);
% %     estimated_page_gray = cv.cvtColor(estimated_page, 'BGR2GRAY');
% % 
% %     page_gray = cv.cvtColor(page, 'BGR2GRAY');
% %     page_gray(estimated_page_gray == 0) = 0;
% % 
% %     image_area = size(estimated_page_gray, 1) * size(estimated_page_gray, 2);
% %     average_pixel_difference = sum(sum(estimated_page_gray - page_gray)) / image_area;
% % 
% %     figure, imshow(page);
% %     figure, imshow(prev_page);
% %     waitforbuttonpress
% %   end
% % 
% %   prev_page = page;
% % end
% 
% rmpath('../map-questions');
% rmpath('../perspective-correction');
