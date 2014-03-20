function images = pick_frames(video_path)
  images = {};
  video = cv.VideoCapture(video_path);

  while video.grab()
    image = video.retrieve();
    imshow(image)

    should_save = input('save? ', 's');
    if should_save == 'y'
      images{end + 1} = image;
    end

    for i = 1 : 5
      video.grab();
    end
  end
end
