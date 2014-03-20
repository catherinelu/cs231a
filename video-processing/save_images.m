function save_images(images, prefix)
  for i = 1 : length(images)
    name = [prefix int2str(i) '.jpg'];
    imwrite(images{i}, name);
  end
end
