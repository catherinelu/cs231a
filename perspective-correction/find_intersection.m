function [x, y] = find_intersection(line1, line2)
  % find point and slope of first line
  r1 = line1(1);
  theta1 = line1(2);

  x1 = r1 * cos(theta1);
  y1 = r1 * sin(theta1);

  rise1 = -cos(theta1);
  run1 = sin(theta1);

  % find point and slope of second line
  r2 = line2(1);
  theta2 = line2(2);

  x2 = r2 * cos(theta2);
  y2 = r2 * sin(theta2);

  rise2 = -cos(theta2);
  run2 = sin(theta2);

  % derive intersection point from the following system of equations:
  % x = x1 + run1 * dist1 = x2 + run2 * dist2
  % y = y1 + rise1 * dist1 = y2 + rise2 * dist2
  dist2 = (run1 * (y1 - y2) + rise1 * (x2 - x1)) / (run1 * rise2 - rise1 * run2);
  x = x2 + run2 * dist2;
  y = y2 + rise2 * dist2;
end
