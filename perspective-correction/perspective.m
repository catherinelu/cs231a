close all;
clear all;

image = imread('/Users/karthikv/Desktop/image2.jpg');
figure, imshow(image);

corrected_image = correct_perspective(image, true);
figure, imshow(corrected_image);
