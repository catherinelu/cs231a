close all;
clear all;

tic;

% where page JPEGs are located
% page_dir = '/Users/karthikv/Dropbox/Shared/CS231A/scoryst-project/cs221-exams-better';
% prefixes = {'adam', 'ahmed', 'alp', 'andy', 'arushi', 'brett'};

page_dir = '/Users/karthikv/Active/scoryst/scraped';
% prefixes = {'adam_weitz_goldberg', 'ai_jiang', 'alfred_zong', 'alice_su-chin_yeh', 'allen_chen', 'ana_klimovic', 'andrew_dotey_jr', 'andrew_yoo_mah', 'anil_das', 'anthony_joseph_mainero', 'aparna_guruprasad_bhat', 'apekshit_sharma', 'arjun_gopalan', 'ashish_gupta', 'ashley_jin', 'ba_quan_truong', 'benjamin_au', 'blank', 'bonnie_joyce_mclindon'};
% prefixes = {'adam_weitz_goldberg', 'adriano_quiroga_garafulic', 'ai_jiang', 'alfred_zong', 'alice_su-chin_yeh', 'allen_chen', 'ana_klimovic', 'andrew_dotey_jr', 'andrew_yoo_mah', 'anil_das', 'anthony_joseph_mainero', 'aparna_guruprasad_bhat', 'apekshit_sharma', 'arjun_gopalan', 'ashish_gupta', 'ashley_jin', 'ba_quan_truong', 'benjamin_au', 'blank', 'bonnie_joyce_mclindon', 'brett_cooley', 'brian_lam', 'brian_lao', 'bryan_michael_lewandowski', 'bryson_kalani_mcfeeley', 'cheng_chen', 'chiara_hordatt_brosco', 'chinmay_kulkarni', 'chinmayee_yogesh_shah', 'chirag_rajesh_sangani', 'chris_stephan_van_harmelen', 'chuan_huang', 'clarence_wen_han_chio', 'clint_john_riley', 'collin_shing-tsi_lee', 'conrad_chan', 'curran_hong_kaushik', 'dan_li', 'dan_thompson', 'daniel_benjamin_jackoway', 'dave_andrew_dolben', 'diego_rafael_moreno_ferrer', 'divya_konda', 'elmer_le', 'emad_elwany', 'eric_michael_yurko', 'evan_samuel_plotkin', 'forrest_ray_browning', 'haowei_zhang', 'haoxiang_zhao', 'haozhun_jin', 'heng_xiao', 'hsin_fang_wu', 'hung_tan_tran', 'ian_walsh', 'iru_wang', 'jagadish_venkatraman', 'james_abraham_shapiro', 'jeff_wheeler', 'jeffrey_ericson', 'jessica_michelle_tai', 'jiaji_hu', 'jiayuan_ma', 'jim_zichao_zheng', 'jimmy_du', 'joanna_kim', 'john_carl_burke', 'johnson_anoop_kochummen', 'jona_babi', 'jonathan_grey_swenson', 'jonathan_yi_hung', 'jorge_aguirre_jr.', 'joseph_baena', 'joseph_tsai', 'juliana_marie_cook', 'kate_christine_stuckman', 'ken_kao', 'kevin_andrew_smith', 'kevin_thomas_durfee', 'khalil_shambe_griffin', 'kristi_elisabeth_bohl', 'kunle_michael_oyedele', 'laura_garrity', 'laura_jade_griffiths', 'lauren_rachel_moser', 'laza_upatising', 'lennon_keagan_jk_chimbumu', 'ling-ling_samantha_zhang', 'lisa_yan', 'lu_yuan', 'lynne_schleiffarth_burks', 'm.j._ma', 'mahmoud_a_r_aly_ragab', 'manni_luca_giovanni_cavalli-sforza', 'marco_antonio_alban-hidalgo', 'marlon_suyo', 'meredith_grace_marks', 'michael_alexander_lublin', 'michael_howard_percy', 'michael_ryan_fitzpatrick', 'mike_logan_jermann', 'narek_tovmasyan', 'nathaniel_jacob_eidelson', 'nicholas_alexandros_platias', 'nicole_hu', 'omar_sebastian_diab', 'omosola_odebunmi_odetunde', 'osbert_bastani', 'pallavi_gupta', 'paul_benigeri', 'perth_charernwattanagul', 'peter_hayes', 'peter_hu', 'peter_william_johnston', 'prachetaa_raghavan', 'quentin_kenneth_moy', 'rajesh_raghavan', 'reid_allen_watson', 'richard_wei_hsu', 'robert_wood_dunlevie', 'roger_chen', 'roneil_parker_rumburg', 'sam_keller', 'sammy_el_ghazzal', 'saurabh_sharan', 'shaurya_saluja', 'sheila_ramaswamy', 'sheta_chatterjee', 'sophia_westwood', 'srinidhi_ramesh_kondaji', 'stephanie_anne_nicholson', 'stephen_michael_barber', 'stephen_walter_trusheim', 'stephen_yang', 'sumedh_rajendra_sawant', 'sumi_narayanan', 'sundaram_ananthanarayanan', 'tejas_shah', 'tina_yu_jin_roh', 'tony_michael_vivoli', 'tony_zhang', 'travis_addair', 'truman_cranor', 'vien_trong_dinh', 'will_henry_thomas_iv', 'xiaofei_fu', 'xiaoye_liu', 'xiaoying_shen', 'xuening_liu', 'xueqian_jiang', 'yao_xiao', 'yeskendir_kassenov', 'yifan_mai', 'yifei_huang', 'yonathan_perez', 'yoon-suk_han', 'yuchi_liu', 'yunfan_yu', 'yutian_liu', 'zheng_wu', 'zi_ling'};
prefixes = {'ken_kao', 'perth_charernwattanagul', 'michael_howard_percy', 'kate_christine_stuckman', 'joseph_baena', 'paul_benigeri', 'zi_ling', 'reid_allen_watson', 'peter_hayes', 'andrew_yoo_mah'};
num_prefixes = length(prefixes);

% page suffixes
pages = 1 : 11;
num_pages = length(pages);

images = cell(num_prefixes, num_pages);
image_features = cell(num_prefixes, num_pages);
image_descriptors = cell(num_prefixes, num_pages);

detector_name = 'SIFT';
extractor_name = 'ORB';

detector = cv.FeatureDetector(detector_name);
extractor = cv.DescriptorExtractor(extractor_name);
matcher = cv.DescriptorMatcher('BruteForce');

% read real exam images, compute SIFT features/descriptors
for i = 1 : num_prefixes
  for j = 1 : num_pages
    path = [page_dir '/' prefixes{i} int2str(pages(j)) '.jpg'];
    % images{i}{j} = im2single(imread(path));
    images{i}{j} = imread(path);

    % convert to grayscale if necessary
    if size(images{i}{j}, 3) == 3
      images{i}{j} = rgb2gray(images{i}{j});
    end

    % resize image to width of 500
    image_size = size(images{i}{j});
    images{i}{j} = imresize(images{i}{j}, 500 / image_size(2));

    % [features, descriptors] = vl_sift(images{i}{j});
    features = detector.detect(images{i}{j});
    descriptors = extractor.compute(images{i}{j}, features);

    image_features{i}{j} = features;
    image_descriptors{i}{j} = descriptors;

    fprintf('Finished computing %s features and %s descriptors for prefix %s, page %g\n', ...
      detector_name, extractor_name, prefixes{i}, pages(j));
  end
end

blank_prefix = 'blank';
blank_images = cell(num_pages);
blank_image_features = cell(num_pages);
blank_image_descriptors = cell(num_pages);

% read blank exam images, compute SIFT keypoints
for i = 1 : num_pages
  path = [page_dir '/' blank_prefix int2str(pages(i)) '.jpg'];
  % blank_images{i} = im2single(imread(path));
  blank_images{i} = imread(path);

  % convert to grayscale if necessary
  if size(blank_images{i}, 3) == 3
    blank_images{i} = rgb2gray(blank_images{i});
  end

  % resize image to width of 500
  image_size = size(blank_images{i});
  blank_images{i} = imresize(blank_images{i}, 500 / image_size(2));

  % [features, descriptors] = vl_sift(blank_images{i});
  features = detector.detect(blank_images{i});
  descriptors = extractor.compute(blank_images{i}, features);

  blank_image_features{i} = features;
  blank_image_descriptors{i} = descriptors;

  fprintf('Finished computing %s features and %s descriptors for blank page %g\n', ...
    detector_name, extractor_name, pages(i));
end

% statistics to report
total_positives = num_prefixes * num_pages;
total_negatives = num_prefixes * num_pages * num_pages - total_positives;
true_positives = 0;
true_negatives = 0;

for i = 1 : num_prefixes
  for j = 1 : num_pages
    consideration_set = zeros(1, num_pages);

    for k = 1 : num_pages
      blank_features = blank_image_features{k};
      blank_descriptors = blank_image_descriptors{k};

      features = image_features{i}{j};
      descriptors = image_descriptors{i}{j};

      % match descriptors and filter to best matches
      matches = matcher.match(blank_descriptors, descriptors);
      distances = [matches.distance];
      best_blank_features = blank_features([matches.queryIdx] + 1);
      best_features = features([matches.trainIdx] + 1);

      [min_distances, indexes] = sort(distances);
      if length(indexes) > 50
        indexes = indexes(1:50);
      end

      best_blank_features = best_blank_features(indexes);
      best_features = best_features(indexes);

      best_blank_points = ones(3, length(indexes));
      best_points = ones(3, length(indexes));
      for l = 1 : length(indexes)
        best_blank_points(1:2, l) = best_blank_features(l).pt;
        best_points(1:2, l) = best_features(l).pt;
      end

      % [matches, scores] = vl_ubcmatch(blank_descriptors, descriptors);
      % best_matches = matches(:, scores < 5000);

      % best_blank_features = blank_features(:, best_matches(1, :));
      % best_features = features(:, best_matches(2, :));
      % num_features = size(best_features, 2);

      % convert keypoints to homogeneous coordinates
      % best_blank_features = [best_blank_features(1, :); best_blank_features(2, :); ...
      %   ones(1, num_features)];
      % best_features = [best_features(1, :); best_features(2, :); ones(1, num_features)];

      % find homography H such that H * best_blank_features = best_features
      % [H, num_inliers, score] = estimate_homography(best_blank_features, best_features, 100);
      [H, num_inliers, score] = estimate_homography(best_blank_points, best_points, 100);

      % compute scale and rotation from homography
      scale = sqrt(H(1, 1) ^ 2 + H(1, 2) ^ 2);
      rotation = acos(H(1, 1) / scale);

      % debugging information:
      % H
      % scale
      % rotation
      % num_features
      % num_inliers

      % if scale is close to 1 and rotation is close to 0, we've found a valid match;
      % add it to the consideration set
      if abs(scale - 1) < 0.3 && abs(rotation) < 0.3 && num_inliers > 0
        % Index represents the blank page index. Value represents how closely the
        % blank page matches the current image. We base the value solely on how
        % many inliers were captured; the more inliers, the more the blank page
        % matched the image.
        consideration_set(k) = num_inliers;
      end
    end

    % find the best score and page in the consideration set
    [best_score, best_page] = max(consideration_set);
    if best_score == 0
      % no best score because nothing matched; invalidate the best page
      best_page = -1;
    end

    % report results
    for k = 1 : num_pages
      if k == best_page
        if j ~= k
          % failed test if pages are actually different
          result = 'FAIL';
        else
          result = 'pass';
          true_positives = true_positives + 1;
        end

        fprintf('[%s] %s page %g matches blank page %g\n', result, prefixes{i}, ...
          pages(j), pages(k));
      else
        if j == k
          % failed test if pages are actually the same
          result = 'FAIL';
        else
          result = 'pass';
          true_negatives = true_negatives + 1;
        end

        fprintf('[%s] %s page %g does NOT match blank page %g\n', result, prefixes{i}, ...
          pages(j), pages(k));
      end
    end
  end
end

toc

% report statistics
fprintf('\n---\n\n')
fprintf('True positives: %g/%g = %g\n', true_positives, total_positives, ...
  true_positives / total_positives);
fprintf('True negatives: %g/%g = %g\n', true_negatives, total_negatives, ...
  true_negatives / total_negatives);
