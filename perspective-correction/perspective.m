close all;
clear all;

image = imread('./page.jpg');
figure, imshow(image);

corrected_image = correct_perspective(image, true);
figure, imshow(corrected_image);
