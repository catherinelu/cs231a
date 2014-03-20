clear all;
close all;

addpath('../perspective-correction');
NUM_FRAMES_TO_SKIP = 10;

video = cv.VideoCapture('./lecture.mov');
pages = {};

% the previous image taken from the video
previous_image = false;

num_frames_for_stability = 10;
num_frames_valid = 0;

frame = 1;


while video.grab()
  image = video.retrieve();

  if previous_image == false
    % no previous image; this must be the first page
    pages{end + 1} = image;
  else
    M = cv.estimateRigidTransform(image, previous_image);

    % Look for when we can't make an affine transform from the current frame to
    % the previous frame. This implies that a page is being flipped (out of
    % plane transformation is not affine). Take the frame that comes after
    % a streak of invalid transformations, as this will represent the next page
    % when flipping is complete.
    if size(M, 1) == 0
      'invalid!'
      num_frames_valid = 0;
    else
      num_frames_valid = num_frames_valid + 1;

      if num_frames_valid == num_frames_for_stability
        'valid stable'
        pages{end + 1} = image;
      end
    end
  end

  previous_image = image;

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
  figure, imshow(corrected_image);
end

rmpath('../perspective-correction');
