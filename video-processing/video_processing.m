clear all;
close all;

VIDEO_PATH = './paper-vertical.mov';
video = cv.VideoCapture(VIDEO_PATH);
pages = {};

num_frames = 0;
differences = [0];

previous_image = false;

while video.grab()
  image = video.retrieve();
  image_gray = cv.cvtColor(image, 'BGR2GRAY');

  if size(previous_image, 1) > 1
    M = cv.estimateRigidTransform(previous_image, image);

    if size(M, 1) == 0
      differences = [differences differences(end)];
    else
      estimated_page = cv.warpAffine(previous_image, M);
      estimated_page_gray = cv.cvtColor(estimated_page, 'BGR2GRAY');

      page_gray = cv.cvtColor(image, 'BGR2GRAY');
      page_gray(estimated_page_gray == 0) = 0;

      image_area = size(estimated_page_gray, 1) * size(estimated_page_gray, 2);
      differences = [differences sum(sum(estimated_page_gray - page_gray))];
    end
  end

  previous_image = image;
  num_frames = num_frames + 1;
end

max_difference = max(differences);
normalized_differences = differences / max_difference;

figure, plot(1 : num_frames, normalized_differences);
grid on;

thresholded = zeros(1, length(normalized_differences));

frames_to_converge = 15;
num_converging_frames = frames_to_converge;

for i = 1 : length(normalized_differences)
  if normalized_differences(i) > 0.35
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

% figure, plot(1 : num_frames, thresholded);
% grid on;

% reset video
video = cv.VideoCapture(VIDEO_PATH);
index = 1;

for frame = 1 : num_frames
  video.grab();

  if frame == frames_to_pick(index)
    page = video.retrieve();
    figure, imshow(page);

    index = index + 1;
    if index > length(frames_to_pick)
      break
    end
  end
end
