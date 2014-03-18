close all;
clear all;

directory = './pages'
page_paths = dir([directory '/*.jpg']);

correct_extractions = [];
incorrect_extractions = [];

for i = 1 : length(page_paths)
  path = [directory '/' page_paths(i).name];
  page = imread(path);

  corrected = correct_perspective(page);
  close all;

  imshow(page);
  figure, imshow(corrected);

  correct = input('correct? ', 's');

  if correct == 'y'
    correct_extractions = [correct_extractions i];
  else
    incorrect_extractions = [incorrect_extractions i];
  end
end

num_correct = length(correct_extractions)
num_total = length(page_paths)
num_correct / num_total
