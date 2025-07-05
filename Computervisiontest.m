% Determine the script directory and the images directory
scriptDir   = fileparts(mfilename('fullpath'));
folder      = fullfile(scriptDir, 'Datasets', 'Frauenkirche/');
% Check if the image was loaded successfully
if isempty(Iorig)
    error('The image %s could not be loaded.', imageFiles(k).name);
end

% Check if the transformation is valid
if k > 1 && isempty(tforms(k).T)
    error('The transformation for image %d is invalid.', k);
end

% Check if the cropped image is not empty
if isempty(Icrop)
    warning('The cropped image for %s is empty and will not be saved.', imageFiles(k).name);
else
    imwrite(Icrop, fullfile(croppedDir, imageFiles(k).name));
end
filePattern = fullfile(folder, '*.jpg');  % only 2D RGB images
imageFiles  = dir(filePattern);
[~, idx]    = sort({imageFiles.name});
imageFiles  = imageFiles(idx);
numImages   = numel(imageFiles);

% 1. Read reference image and prepare grayscale version
tmp      = imread(fullfile(folder, imageFiles(1).name));
IrefRGB  = tmp;
IrefGray = im2double(rgb2gray(IrefRGB));

% --- NEW: Calculate ROI for SURF (top 85 % of the image) ---
[refH, refW] = size(IrefGray);
cutoffRow    = floor(refH * 0.90);         % Row up to which we apply SURF
surfROI      = [1, 1, refW, cutoffRow];    % [x, y, width, height]

% 2. Define output reference for all warps
outputView = imref2d(size(IrefGray));

% 3. Initialization: transformations array and global mask
tforms     = repmat(affine2d(eye(3)), 1, numImages);

% 4. Loop: Estimate transformations
for k = 2:numImages
    % 4.1 Load image & histogram matching
    I2RGB     = imread(fullfile(folder, imageFiles(k).name));
    I2matched = zeros(size(I2RGB), 'like', I2RGB);
    for c = 1:3
        I2matched(:,:,c) = imhistmatch(I2RGB(:,:,c), IrefRGB(:,:,c));
    end
    I2gray = im2double(rgb2gray(I2matched));
    
    % 4.2 SURF feature matching only in upper image half (85 %)
    pts1 = detectSURFFeatures(IrefGray, 'ROI', surfROI, ...
    'MetricThreshold', 200, ...
    'NumOctaves', 6, ...
    'NumScaleLevels', 10);

    pts2 = detectSURFFeatures(I2gray, 'ROI', surfROI, ...
    'MetricThreshold', 200, ...
    'NumOctaves', 6, ...
    'NumScaleLevels', 10);
    
    [f1, vpts1] = extractFeatures(IrefGray, pts1);
    [f2, vpts2] = extractFeatures(I2gray,    pts2);
    idxPairs    = matchFeatures(f1, f2, 'Unique', true);
    matched1    = vpts1(idxPairs(:,1));
    matched2    = vpts2(idxPairs(:,2));
    
    % Estimate similarity transformation
    tforms(k) = estimateGeometricTransform(matched2, matched1, ...
                   'similarity', 'MaxDistance', 4, ...
                   'Confidence',  99.9, ...
                   'MaxNumTrials',3000);
end

% === Step 5: Automatic determination of the common non-black area ===
BB = zeros(numImages,4);  % [xmin, ymin, width, height]
for k = 1:numImages
    % 5.1 Load image, match & warp
    Iorig = imread(fullfile(folder, imageFiles(k).name));
    if k > 1
        Itemp = zeros(size(Iorig), 'like', Iorig);
        for c = 1:3
            Itemp(:,:,c) = imhistmatch(Iorig(:,:,c), IrefRGB(:,:,c));
        end
        Iw = imwarp(Itemp, tforms(k), 'OutputView', outputView);
    else
        Iw = Iorig;
    end
    
    % 5.2 Create mask of all non-black pixels
    mask = any(Iw~=0,3);
    
    % 5.3 Determine bounding box of this mask
    s = regionprops(mask, 'BoundingBox');
    BB(k,:) = s.BoundingBox;
end

% 5.4 Calculate common crop
x0 = max(BB(:,1));
y0 = max(BB(:,2));
x1 = min(BB(:,1) + BB(:,3));
y1 = min(BB(:,2) + BB(:,4));
w  = x1 - x0;
h  = y1 - y0;
rect = [ floor(x0)+1, floor(y0)+1, floor(w), floor(h) ];

% 5a. Create output folder for cropped images
croppedDir = fullfile(folder, 'common_crop');
if ~exist(croppedDir, 'dir')
    mkdir(croppedDir);
end

% === Step 6: Crop and save all images ===
for k = 1:numImages
    Iorig = imread(fullfile(folder, imageFiles(k).name));
    if k > 1
        Itemp = zeros(size(Iorig), 'like', Iorig);
        for c = 1:3
            Itemp(:,:,c) = imhistmatch(Iorig(:,:,c), IrefRGB(:,:,c));
        end
        Iw = imwarp(Itemp, tforms(k), 'OutputView', outputView);
    else
        Iw = Iorig;
    end
    
    Icrop = imcrop(Iw, rect);
    imwrite(Icrop, fullfile(croppedDir, imageFiles(k).name));
end

% === Step 7: Comparison and display loop (optional) ===
for k = 2:numImages
    I2RGB     = imread(fullfile(folder, imageFiles(k).name));
    I2matched = zeros(size(I2RGB), 'like', I2RGB);
    for c = 1:3
        I2matched(:,:,c) = imhistmatch(I2RGB(:,:,c), IrefRGB(:,:,c));
    end
    I2regRGB = imwarp(I2matched, tforms(k), 'OutputView', outputView);
    
    I1c = imcrop(IrefRGB, rect);
    I2c = imcrop(I2regRGB, rect);
    
    % --- Commented out: difference and red overlay ---
    % G1        = im2double(rgb2gray(I1c));
    % G2        = im2double(rgb2gray(I2c));
    % diff      = abs(G2 - G1);
    % threshold = 0.2;
    % changeMask= diff > threshold;
    % redMask   = uint8(cat(3, changeMask, zeros(size(changeMask)), zeros(size(changeMask)))) * 255;
    % overlay   = imadd(I2c, redMask);
    
    figure('Name', sprintf('Comparison Ref → Image %d', k), 'NumberTitle', 'off');
    subplot(1,3,1), imshow(I1c),    title(['Ref: '   imageFiles(1).name], 'Interpreter', 'none');
    subplot(1,3,2), imshow(I2c),    title(['Mov: '   imageFiles(k).name], 'Interpreter', 'none');
    %subplot(1,3,3), imshow(overlay), title('Changes (red)'); % <--- Commented out
end

% Montage of all common crops
cropFiles = dir(fullfile(croppedDir, '*.jpg'));
numCrop   = numel(cropFiles);
cols      = ceil(sqrt(numCrop));
rows      = ceil(numCrop/cols);

figure('Name','All common crops','NumberTitle','off');
for k = 1:numCrop
    I = imread(fullfile(croppedDir, cropFiles(k).name));
    subplot(rows, cols, k);
    imshow(I);
    title(cropFiles(k).name, 'Interpreter', 'none', 'FontSize', 8);
    axis off;
end
% 4.3a Visualization of SURF points in upper 85 %
    figure('Name', sprintf('SURF points image %d', k), 'NumberTitle', 'off');
    
    % Left half: reference image
    subplot(1,2,1);
    imshow(IrefGray);
    title('Reference (upper 85 %)');
    hold on;
    % only show strongest 100 points (you can adjust this)
    strongest1 = pts1.selectStrongest(100);
    plot(strongest1);
    % ROI as green box
    rectangle('Position', surfROI, 'EdgeColor', 'g', 'LineWidth', 1);
    hold off;
    
    % Right half: current image
    subplot(1,2,2);
    imshow(I2gray);
    title(sprintf('Image %d: %s', k, imageFiles(k).name), 'Interpreter', 'none');
    hold on;
    strongest2 = pts2.selectStrongest(100);
    plot(strongest2);
    rectangle('Position', surfROI, 'EdgeColor', 'g', 'LineWidth', 1);
    hold off;
