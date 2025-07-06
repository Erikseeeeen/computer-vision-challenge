%% Main Script: Register, crop, and display batch images

% 1. Determine script and images directory
scriptDir = fileparts(mfilename('fullpath'));
folder    = fullfile(scriptDir, 'Datasets', 'Kuwait/');

% 2. Gather all .jpg and .JPG files, sort alphabetically
files1 = dir(fullfile(folder, '*.jpg'));
files2 = dir(fullfile(folder, '*.JPG'));
imageFiles = [files1; files2];
if isempty(imageFiles)
    error('No images found in folder "%s".', folder);
end
[~, idx] = sort({imageFiles.name});
imageFiles = imageFiles(idx);
numImages = numel(imageFiles);

% 3. Read reference image (first one) and prepare grayscale version
IrefRGB  = imread(fullfile(folder, imageFiles(1).name));
IrefGray = im2double(rgb2gray(IrefRGB));
IrefGray = im2double(rgb2gray(IrefRGB));
IrefGray = adapthisteq(IrefGray, 'ClipLimit',0.02, 'NumTiles',[8 8]);
IrefGray = imadjust(IrefGray, stretchlim(IrefGray,[0.01 0.99]));
IrefGray = IrefGray .^ 0.8;
IrefGray = imgaussfilt(IrefGray, 1);
IrefGray = imsharpen(IrefGray, 'Radius',1, 'Amount',1);
% 4. Define SURF ROI (top 90% of the image)
[refH, refW] = size(IrefGray);
cutoffRow    = floor(refH * 0.90);
surfROI      = [1, 1, refW, cutoffRow];

% 5. Initialize transformation array and output reference
outputView = imref2d(size(IrefGray));
tforms     = repmat(affine2d(eye(3)), 1, numImages);

% 6. Loop: Estimate transformations for all images (except reference)
for k = 2:numImages
    I2RGB     = imread(fullfile(folder, imageFiles(k).name));
    I2matched = histMatchToRef(I2RGB, IrefRGB);
    I2gray    = im2double(rgb2gray(I2matched));
    I2RGB   = imread(fullfile(folder, imageFiles(k).name));
I2match = histMatchToRef(I2RGB, IrefRGB);

% Graustufen
I2gray = im2double(rgb2gray(I2match));

% 1) Lokales CLAHE
I2gray = adapthisteq(I2gray, 'ClipLimit',0.02, 'NumTiles',[8 8]);

% 2) Globales Strecken
I2gray = imadjust(I2gray, stretchlim(I2gray,[0.01 0.99]));

% 3) Gamma-Korrektur
I2gray = I2gray .^ 0.8;

% 4) Glätten + Schärfen
I2gray = imgaussfilt(I2gray, 1);
I2gray = imsharpen(I2gray, 'Radius',1, 'Amount',1);
    pts1 = detectSURFFeatures(IrefGray, 'MetricThreshold', 100);
    pts2 = detectSURFFeatures(I2gray,    'MetricThreshold', 100);
    
    [f1, vpts1] = extractFeatures(IrefGray, pts1);
    [f2, vpts2] = extractFeatures(I2gray,    pts2);
    idxPairs    = matchFeatures(f1, f2, 'Unique', true);
    matched1    = vpts1(idxPairs(:,1));
    matched2    = vpts2(idxPairs(:,2));
    
    if isempty(idxPairs)
        error('No matching features found between image %d and reference.', k);
    end
    
    tforms(k) = estimateGeometricTransform(...
    matched2, matched1, ...
    'similarity', ...      % statt 'projective'
    'MaxDistance',   12, ...    % erlaubt größere Abweichungen
    'Confidence',    99, ...    % stoppt früher
    'MaxNumTrials', 5000);      % mehr Versuche
end

% 7. Automatic determination of the common crop region (bounding box)
BB = zeros(numImages,4);  % [xmin, ymin, width, height]
for k = 1:numImages
    Iorig = imread(fullfile(folder, imageFiles(k).name));
    if k > 1
        Itemp = histMatchToRef(Iorig, IrefRGB);
        Iw = imwarp(Itemp, tforms(k), 'OutputView', outputView);
    else
        Iw = Iorig;
    end
    mask = any(Iw~=0,3);
    s = regionprops(mask, 'BoundingBox');
    if isempty(s)
        error('Image "%s" is empty after transformation.', imageFiles(k).name);
    end
    BB(k,:) = s(1).BoundingBox;
end

x0 = max(BB(:,1));
y0 = max(BB(:,2));
x1 = min(BB(:,1) + BB(:,3));
y1 = min(BB(:,2) + BB(:,4));
w  = x1 - x0;
h  = y1 - y0;
rect = [ floor(x0)+1, floor(y0)+1, floor(w), floor(h) ];

% 8. Create output directory for crops
croppedDir = fullfile(folder, 'common_crop');
if ~exist(croppedDir, 'dir')
    mkdir(croppedDir);
end

% 9. Crop and save all images
for k = 1:numImages
    Iorig = imread(fullfile(folder, imageFiles(k).name));
    if k > 1
        Itemp = histMatchToRef(Iorig, IrefRGB);
        Iw    = imwarp(Itemp, tforms(k), 'OutputView', outputView);
    else
        Iw    = Iorig;
    end
    
    % Berechne Crop-Rechteck
    [hIw, wIw, ~] = size(Iw);
    xEnd = min(rect(1)+rect(3)-1, wIw);
    yEnd = min(rect(2)+rect(4)-1, hIw);
    cropRect = [rect(1), rect(2), xEnd-rect(1), yEnd-rect(2)];
    
    try
        Icrop = imcrop(Iw, cropRect);
        if isempty(Icrop)
            warning('Leerer Crop für "%s". Fallback aufs Originalbild.', imageFiles(k).name);
            Icrop = Iorig;
        end
    catch ME
        warning('Crop-Fehler bei "%s": %s\nFallback aufs Originalbild.', imageFiles(k).name, ME.message);
        Icrop = Iorig;
    end
    
    % Speichern
    outPath = fullfile(croppedDir, imageFiles(k).name);
    imwrite(Icrop, outPath);
end


% 10. Comparison and display loop (optional)
for k = 2:numImages
    I2RGB     = imread(fullfile(folder, imageFiles(k).name));
    I2matched = histMatchToRef(I2RGB, IrefRGB);
    I2regRGB  = imwarp(I2matched, tforms(k), 'OutputView', outputView);
    I1c = imcrop(IrefRGB, rect);
    I2c = imcrop(I2regRGB, rect);
    figure('Name', sprintf('Comparison Ref → Image %d', k), 'NumberTitle', 'off');
    subplot(1,2,1), imshow(I1c), title(['Ref: ' imageFiles(1).name], 'Interpreter', 'none');
    subplot(1,2,2), imshow(I2c), title(['Mov: ' imageFiles(k).name], 'Interpreter', 'none');
end

% 11. Show all common crops as montage
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

%{ 12. SURF points visualization loop
for k = 2:numImages
    I2RGB     = imread(fullfile(folder, imageFiles(k).name));
    I2matched = histMatchToRef(I2RGB, IrefRGB);
    I2gray    = im2double(rgb2gray(I2matched));
    pts1 = detectSURFFeatures(IrefGray, 'ROI', surfROI, ...
        'MetricThreshold', 200, 'NumOctaves', 6, 'NumScaleLevels', 10);
    pts2 = detectSURFFeatures(I2gray, 'ROI', surfROI, ...
        'MetricThreshold', 200, 'NumOctaves', 6, 'NumScaleLevels', 10);

    %figure('Name', sprintf('SURF points Image %d', k), 'NumberTitle', 'off');
    %subplot(1,2,1);
    %imshow(IrefGray); title('Reference (top 90%)');
    %hold on;
    %strongest1 = pts1.selectStrongest(100);
    %plot(strongest1);
    %rectangle('Position', surfROI, 'EdgeColor', 'g', 'LineWidth', 1);
    %hold off;
    %subplot(1,2,2);
    %imshow(I2gray); title(sprintf('Image %d: %s', k, imageFiles(k).name), 'Interpreter', 'none');
    %hold on;
    %strongest2 = pts2.selectStrongest(100);
    %plot(strongest2);
    %rectangle('Position', surfROI, 'EdgeColor', 'g', 'LineWidth', 1);
    %hold off;
end

%% Helper function: Histogram matching of RGB images
function Iout = histMatchToRef(Iin, Iref)
    Iout = zeros(size(Iin), 'like', Iin);
    for c = 1:size(Iin,3)
        Iout(:,:,c) = imhistmatch(Iin(:,:,c), Iref(:,:,c));
    end
end