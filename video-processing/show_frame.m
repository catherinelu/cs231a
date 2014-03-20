function image = show_frame(video_path, frame)
  video = cv.VideoCapture(video_path);
  for i = 1 : frame
    if ~video.grab()
      fprintf('warning: end of video!\n')
    end
  end

  image = video.retrieve();
  figure, imshow(image);
end
