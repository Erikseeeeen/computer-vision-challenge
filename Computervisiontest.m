% Advanced 2D Satellite Image Registration and Robust Change Overlay
% Implements coarse-to-fine feature registration and robust change overlay
% Excludes bottom-left 10% region from comparisons

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
ransacMaxDist    = 2;       % RANSAC max distance
ransacConfidence = 99.99;   % RANSAC confidence (%)
ransacMaxTrials  = 5000;    % RANSAC trials

% Change overlay parameters
changeThreshold  = 0.2;     % Gray-level difference threshold
morphRadius      = 2;       % Radius for morphological cleanup
ignoreFractionX  = 0.10;    % Exclude left 10% of width
ignoreFractionY  = 0.10;    % Exclude bottom 10% of height

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

    %--- Robust transform estimation ---
    if size(idxPairs,1) < 4
        tform = projective2d(eye(3));  % Identity fallback
    else
        matched1 = v1(idxPairs(:,1));
        matched2 = v2(idxPairs(:,2));
        tform = estimateGeometricTransform2D(matched2.Location, matched1.Location, ...
            'projective','MaxDistance',ransacMaxDist,'Confidence',ransacConfidence,'MaxNumTrials',ransacMaxTrials);
    end

    %--- Warp moving image to reference frame ---
    refSize = size(Gref);
    outputRef = imref2d(refSize);
    Iwarp = imwarp(Imov, tform, 'OutputView', outputRef);
    Gwarp = rgb2gray(Iwarp);

    %--- Exclude bottom-left region from mask ---
    [h,w] = size(Gref);
    maskValid = true(h,w);
    xIgnore = 1:floor(ignoreFractionX * w);
    yIgnore = ceil((1 - ignoreFractionY) * h):h;
    maskValid(yIgnore,xIgnore) = false;

    %--- Compute difference and apply threshold ---
    diffMap = abs(Gwarp - Gref);
    changeMask = (diffMap > changeThreshold) & maskValid;

    %--- Morphological cleanup for robustness ---
    se = strel('disk', morphRadius);
    cleanMask = imopen(changeMask, se);

    %--- Create red overlay ---
    redLayer = cleanMask;
    overlay = Iwarp;
    overlay(:,:,1) = overlay(:,:,1) + redLayer;
    overlay(:,:,2) = overlay(:,:,2) .* ~redLayer;
    overlay(:,:,3) = overlay(:,:,3) .* ~redLayer;

    %--- Display only the overlay ---
    figure('Name', sprintf('Overlay %d→%d',k,k+1),'NumberTitle','off');
    imshow(overlay); title(sprintf('Detected Changes Overlay %d→%d',k,k+1));
end

%% End of Script
