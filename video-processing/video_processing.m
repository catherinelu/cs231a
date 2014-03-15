clear all;
close all;

addpath('../perspective-correction');
% NUM_FRAMES_TO_SKIP = 1;

video = cv.VideoCapture('./circuits-vertical.mov');
pages = {};

% the previous image taken from the video
previous_image = false;

% whether the last affine transform was invalid
% last_transform_invalid = false;

% number of consecutive frames with a valid transform needed to be deemed stable
num_frames_for_stability = 5;

% number of consecutive frames with a valid transform
num_frames_valid = 0;

frame = 1;

kernel = [0 1 0; 1 -4 1; 0 1 0];

while video.grab()
  image = video.retrieve();

  if previous_image == false
    % no previous image; this must be the first page
    % pages{end + 1} = image;
  else
    M = cv.estimateRigidTransform(image, previous_image);

    % if frame == 305
    %   figure, imshow(image);
    % end

    % start: 315
    % end: 370
    % if frame >= 280 && frame <= 370
    %   M
    %   [frame, num_frames_valid]
    % end

    % image_gray = cv.cvtColor(image, 'BGR2GRAY');
    % filtered = cv.filter2D(image_gray, kernel);
    % blurriness = max(max(filtered));

    % if blurriness < 2
    %   [frame, blurriness]
    %   % figure, imshow(image)
    % end
    blurriness = (sum(sum(cv.Canny(image, [50 200])))) / (size(image, 1) * size(image, 2));

    if blurriness < 1.5
      'invalid'
      num_frames_valid = 0;
    else
      num_frames_valid = num_frames_valid + 1;

      if num_frames_valid == num_frames_for_stability
        'stable!'
        pages{end + 1} = image;
      end
    end

    % Look for when we can't make an affine transform from the current frame to
    % the previous frame. This implies that a page is being flipped (out of
    % plane transformation is not affine). Take the frame that comes after
    % a streak of invalid transformations, as this will represent the next page
    % when flipping is complete.
    % if size(M, 1) == 0
    %   'invalid'
    %   % last_transform_invalid = true;
    %   num_frames_valid = 0;
    % else
    %   % last_transform_invalid = false;
    %   num_frames_valid = num_frames_valid + 1;

    %   if num_frames_valid == num_frames_for_stability
    %     'stable!'
    %     pages{end + 1} = image;
    %   end
    % end
  end

  previous_image = image;
  frame = frame + 1;

  % skip NUM_FRAMES_TO_SKIP; we don't need so much information
  % for i = 1 : NUM_FRAMES_TO_SKIP - 1
  %   if ~video.grab()
  %     break
  %   end
  % end
end

% apply perspective correction to the images
for i = 1 : length(pages)
  corrected_image = correct_perspective(pages{i});

  if corrected_image ~= false
    figure, imshow(corrected_image);
  end
end

rmpath('../perspective-correction');
