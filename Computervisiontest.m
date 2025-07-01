% Advanced 2D Satellite Image Registration and Robust Change Overlay
% Implements coarse-to-fine feature registration, robust change overlay,
% excludes bottom 15% region for change detection, but shows full image including the lower border

%% 1. Settings and Parameters
folder           = 'C:/Users/jhall/OneDrive/Documents/GitHub/computer-vision-challenge/Datasets/Brazilian Rainforest';  % Input folder path
filePattern      = fullfile(folder, '*.jpg');  % Process only 2D RGB images
imageFiles       = dir(filePattern);
[~, idx]         = sort({imageFiles.name});
imageFiles       = imageFiles(idx);
numImages        = numel(imageFiles);

% Feature matching parameters
featureDetector  = 'KAZE';  % 'SIFT' or 'KAZE'
matchMaxRatio    = 0.7;     % Lowe's ratio test threshold
ransacMaxDist    = 2;       % RANSAC max distance (pixels)
ransacConfidence = 99.99;   % RANSAC confidence (%)
ransacMaxTrials  = 5000;    % RANSAC trials

% Change overlay parameters
changeThreshold  = 0.2;     % Gray-level difference threshold
morphRadius      = 2;       % Radius for morphological cleanup
ignoreFractionY  = 0.15;    % Exclude bottom 15% of height for change detection

%% 2. Processing Loop
for k = 1:numImages-1
    %--- Load reference and moving images ---
    Iref = im2double(imread(fullfile(folder, imageFiles(k).name)));
    Imov = im2double(imread(fullfile(folder, imageFiles(k+1).name)));

    %--- Convert to grayscale ---
    Gref = rgb2gray(Iref);
    Gmov = rgb2gray(Imov);

    %--- Feature detection & description ---
    if strcmp(featureDetector,'SIFT')
        pts1 = detectSIFTFeatures(Gref,'ContrastThreshold',0.01);
        pts2 = detectSIFTFeatures(Gmov,'ContrastThreshold',0.01);
    else
        pts1 = detectKAZEFeatures(Gref,'Threshold',0.001);
        pts2 = detectKAZEFeatures(Gmov,'Threshold',0.001);
    end
    [f1,v1] = extractFeatures(Gref,pts1);
    [f2,v2] = extractFeatures(Gmov,pts2);
    idxPairs = matchFeatures(f1,f2,'MaxRatio',matchMaxRatio,'Unique',true);

    %--- Collect matched points and filter bottom region ---
    matched1 = v1(idxPairs(:,1));
    matched2 = v2(idxPairs(:,2));
    [h,~] = size(Gref);
    yLimit = (1 - ignoreFractionY) * h;
    validIdx = matched1.Location(:,2) <= yLimit;
    matched1 = matched1(validIdx);
    matched2 = matched2(validIdx);

    %--- Display filtered matched reference points ---
    figure('Name', sprintf('Matched Features %d→%d',k,k+1),'NumberTitle','off');
    showMatchedFeatures(Gref, Gmov, matched1, matched2, 'montage');
    title(sprintf('Filtered Matched Features %d→%d', k, k+1));

    %--- Robust transform estimation ---
    if numel(matched1) < 4
        tform = projective2d(eye(3));
    else
        tform = estimateGeometricTransform2D(matched2.Location, matched1.Location, ...
            'projective','MaxDistance',ransacMaxDist,'Confidence',ransacConfidence,'MaxNumTrials',ransacMaxTrials);
    end

    %--- Warp moving image to reference frame ---
    refSize = size(Gref);
    outputRef = imref2d(refSize);
    Iwarp = imwarp(Imov, tform, 'OutputView', outputRef);
    Gwarp = rgb2gray(Iwarp);

    %--- Prepare valid mask for change detection only ---
    maskValid = true(h, refSize(2));
    yIgnore = ceil((1 - ignoreFractionY) * h):h;
    maskValid(yIgnore, :) = false;

    %--- Compute difference mask and clean ---
    diffMap = abs(Gwarp - Gref);
    changeMask = (diffMap > changeThreshold) & maskValid;
    se = strel('disk', morphRadius);
    cleanMask = imopen(changeMask, se);

    %--- Create red overlay on full image ---
    overlay = Iwarp;
    overlay(:,:,1) = overlay(:,:,1) + cleanMask;               % Add red where change
    overlay(:,:,2) = overlay(:,:,2) .* ~cleanMask;             % Suppress green
    overlay(:,:,3) = overlay(:,:,3) .* ~cleanMask;             % Suppress blue

    % Note: the lower 15% still shows original image without overlay, and entire image is displayed

    %--- Display full overlay including lower border ---
    figure('Name', sprintf('Overlay %d→%d',k,k+1),'NumberTitle','off');
    imshow(overlay); axis on; title(sprintf('Detected Changes Overlay %d→%d (Full Image)',k,k+1));
end

%% End of Script
