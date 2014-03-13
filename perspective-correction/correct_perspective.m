function corrected_image = correct_perspective(image, debug)
  % debug defaults to false
  if nargin < 2
    debug = false;
  end

  corrected_image = false;
  image_size = size(image);
  image_gray = cv.cvtColor(image, 'BGR2GRAY');

  edges = cv.Canny(image_gray, [50, 200]);
  if debug
    figure, imshow(edges);
  end

  polar_lines_cell = cv.HoughLines(edges, 'Threshold', 150);
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

    lined_image = cv.line(lined_image, [x1 y1], [x2 y2], 'Color', [0 255 0], 'Thickness', 2);
  end

  if debug
    figure, imshow(lined_image);
  end

  iterations = 1000;

  % if the width and height of the bounding box are greater than this value,
  % then we've found a good match
  box_size_threshold = min(image_size(1:2)) / 4;

  max_area = 0;
  best_corners = [];

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
      dimensions = abs([lines(1, 1) - lines(1, 2), lines(1, 3) - lines(1, 4)]);

      if dimensions(1) > box_size_threshold && dimensions(2) > box_size_threshold
        % find four corners of rectangle
        [x1, y1] = find_intersection(lines(:, 1), lines(:, 3));
        [x2, y2] = find_intersection(lines(:, 3), lines(:, 2));
        [x3, y3] = find_intersection(lines(:, 2), lines(:, 4));
        [x4, y4] = find_intersection(lines(:, 4), lines(:, 1));

        % order corners: top left -> bottom left -> top right -> bottom right
        corners = [x1 y1; x2 y2; x3 y3; x4 y4];
        corners = sortrows(corners, 1);
        corners(1:2, :) = sortrows(corners(1:2, :), 2);
        corners(3:4, :) = sortrows(corners(3:4, :), 2);

        height = sqrt(sum((corners(1, :) - corners(2, :)) .^ 2));
        width = sqrt(sum((corners(1, :) - corners(3, :)) .^ 2));

        area = width * height;

        if area > max_area
          max_area = area;
          best_corners = corners;
        end
      end
    end
  end

  if debug
    bounding_box_image = image;
    bounding_box_image = cv.line(bounding_box_image, best_corners(1, :), best_corners(2, :), ...
      'Color', [0 255 0], 'Thickness', 10);
    bounding_box_image = cv.line(bounding_box_image, best_corners(2, :), best_corners(4, :), ...
      'Color', [0 255 0], 'Thickness', 10);
    bounding_box_image = cv.line(bounding_box_image, best_corners(4, :), best_corners(3, :), ...
      'Color', [0 255 0], 'Thickness', 10);
    bounding_box_image = cv.line(bounding_box_image, best_corners(3, :), best_corners(1, :), ...
      'Color', [0 255 0], 'Thickness', 10);
    figure, imshow(bounding_box_image);
  end


  % % find bounding box around rectangle and crop the image
  top_left = [min(best_corners(:, 1)), min(best_corners(:, 2))];
  rectangle_width = max(best_corners(:, 1)) - min(best_corners(:, 1));
  rectangle_height = max(best_corners(:, 2)) - min(best_corners(:, 2));

  cropped_image = image(round(top_left(2)) : round(top_left(2) + rectangle_height), ...
    round(top_left(1)) : round(top_left(1) + rectangle_width));

  if debug
    figure, imshow(cropped_image);
  end

  % correct perspective so rectangle fills the cropped image
  best_corners = best_corners - repmat(top_left, 4, 1);
  cropped_size = size(cropped_image);
  target_corners = [0 0; 0 cropped_size(1); cropped_size(2) 0; ...
    cropped_size(2) cropped_size(1)];

  H = cv.getPerspectiveTransform(best_corners, target_corners);
  corrected_image = cv.warpPerspective(cropped_image, H);
end
