clear all;
close all;

image = im2single(imread('blurred.jpg'));
image_size = size(image);

psf_guess = ones(3, 3);
[deblurred_image, psf] = deconvblind(image, psf_guess, 30);

figure;
imshow(image);

figure;
imshow(deblurred_image);
