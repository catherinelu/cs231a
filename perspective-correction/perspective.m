close all;
clear all;

image = imread('/Users/karthikv/Desktop/image2.jpg');
figure, imshow(image);

image_size = size(image);
image_gray = cv.cvtColor(image, 'BGR2GRAY');

edges = cv.Canny(image_gray, [50, 200]);
figure, imshow(edges);

polar_lines_cell = cv.HoughLines(edges, 'Threshold', 100);
num_lines = length(polar_lines_cell);

% create matrix of polar lines from cell array
polar_lines = zeros(2, num_lines);
lined_image = image;

for i = 1 : num_lines
  polar_line = polar_lines_cell{i};
  polar_lines(:, i) = [polar_line(1); polar_line(2)];

  % draw Hough lines for reference
  r = polar_line(1);
  theta = polar_line(2);

  rise = -cos(theta);
  run = sin(theta);

  x0 = r * cos(theta);
  y0 = r * sin(theta);

  x1 = round(x0 + run * -10000);
  y1 = round(y0 + rise * -10000);
  x2 = round(x0 + run * 10000);
  y2 = round(y0 + rise * 10000);

  lined_image = cv.line(lined_image, [x1 y1], [x2 y2], 'Color', [0 255 0]);
end

figure, imshow(lined_image);
iterations = 10000;

% employ RANSAC to find best square
for i = 1 : iterations
  subset = vl_colsubset(1 : num_lines, 4);
  lines = polar_lines(:, subset);
  theta0 = lines(2, 1);

  % find angles between lines; convert to degrees
  angles = [lines(2, 2) - theta0, lines(2, 3) - theta0, lines(2, 4) - theta0];
  angles = angles * 180 / pi;

  % limit range of angles from [0, 180); a line with theta = -50 is in the same
  % direction as a line with theta = 130
  angles(angles < 0) = mod(angles(angles < 0) + 180, 180);

  % lines extend to infinity, so an angle difference of 170 is the same as an
  % angle difference of 10
  angles(angles > 90) = 180 - angles(angles > 90);

  % sort angles and lines in parallel
  [angles, indexes] = sort(angles);
  lines = lines(:, [1, indexes + 1]);

  if angles(1) < 30 && angles(2) > 60 && angles(3) > 60
    % we have two lines that are parallel, and two others 90 degrees apart; these
    % make a quadrilateral!

    distances = abs([lines(1, 1) - lines(1, 2), lines(1, 3) - lines(1, 4)]);
    if distances(1) > image_size(1) / 6 && distances(2) > image_size(1) / 6
      % find four corners of rectangle
      [x1, y1] = find_intersection(lines(:, 1), lines(:, 3));
      [x2, y2] = find_intersection(lines(:, 3), lines(:, 2));
      [x3, y3] = find_intersection(lines(:, 2), lines(:, 4));
      [x4, y4] = find_intersection(lines(:, 4), lines(:, 1));

      % order corners: top left -> bottom left -> top right -> bottom right
      x_y_pairs = [x1 y1; x2 y2; x3 y3; x4 y4];
      x_y_pairs = sortrows(x_y_pairs, 1);
      x_y_pairs(1:2, :) = sortrows(x_y_pairs(1:2, :), 2);
      x_y_pairs(3:4, :) = sortrows(x_y_pairs(3:4, :), 2);

      % warp rectangle to fill image
      target_pairs = [0 0; 0 image_size(1); image_size(2) 0; image_size(2) image_size(1)];
      H = cv.getPerspectiveTransform(x_y_pairs, target_pairs)
      corrected_image = cv.warpPerspective(image, H);

      figure, imshow(corrected_image);
      break;
    end
  end
end
