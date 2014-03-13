base_image_dir = './images';
base_cropped_dir = './cropped';

letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
for i = 1 : length(letters)
  % find correct directory for lowercase/uppercase character images
  if isstrprop(letters(i), 'lower')
    image_dir = [base_image_dir '/' letters(i) '-lower'];
    cropped_dir = [base_cropped_dir '/' letters(i) '-lower'];
  else
    image_dir = [base_image_dir '/' letters(i)];
    cropped_dir = [base_cropped_dir '/' letters(i)];
  end

  files = dir(image_dir);
  files = files(3:end);  % ignore '.' and '..'

  for j = 1 : length(files)
    image = rgb2gray(imread([image_dir '/' files(j).name]));

    % strip whitespace vertically
    sum_across_cols = sum(image, 2);
    image = image(sum_across_cols < 255 * size(image, 2), :); 

    % strip whitespace horizontally
    sum_across_rows = sum(image, 1);
    image = image(:, sum_across_rows < 255 * size(image, 1));

    imwrite(image, [cropped_dir '/' int2str(j) '.png']);
  end
end
