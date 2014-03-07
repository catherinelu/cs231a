% Estimates the homography H such that H * x = b. Performs RANSAC iterations
% times. Returns H along with the number of inliers and its total score (the
% amount of error; lower is better).
function [best_H, best_num_inliers, best_total_score] = estimate_homography(x, b, iterations)
  threshold = 5;
  best_num_inliers = 0;
  best_total_score = inf;

  num_columns = size(x, 2);
  best_H = zeros(3, 3);

  for j = 1 : iterations
    subset = vl_colsubset(1 : num_columns, 4);
    x_subset = x(:, subset);
    b_subset = b(:, subset);

    % H * x = b implies that x' * H' = b'. Hence, H = (x' \ b')'.
    % Warnings suggest a bad homography, which we'll discard anyway. Suppress them.
    warning('off', 'MATLAB:rankDeficientMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');
    H = (x' \ b')';
    warning('on', 'MATLAB:singularMatrix');
    warning('on', 'MATLAB:nearlySingularMatrix');
    warning('on', 'MATLAB:rankDeficientMatrix');

    H = H / H(3, 3);

    % determine score and inliers
    score = (H * x - b) .^ 2;
    score = sqrt(sum(score, 1));

    inliers = score <= threshold;
    num_inliers = sum(inliers);
    total_score = sum(score(inliers));

    % if we have more inliers, or if we have equal inliers and a better score,
    % this is the new best homography
    if num_inliers > best_num_inliers || (num_inliers == best_num_inliers && ...
        total_score < best_total_score)
      best_H = H;
      best_total_score = total_score;
      best_num_inliers = num_inliers;
    end
  end
end
