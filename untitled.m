%% Main Script: Register, crop, and display batch images

% 1. Determine script and images directory
scriptDir = fileparts(mfilename('fullpath'));
folder    = fullfile(scriptDir, 'Datasets', 'Dubai');

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
    
    try
        % --- SURF-based matching ---
        pts1 = detectSURFFeatures(IrefGray, 'ROI', surfROI, ...
            'MetricThreshold', 200, 'NumOctaves', 6, 'NumScaleLevels', 10);
        pts2 = detectSURFFeatures(I2gray, 'ROI', surfROI, ...
            'MetricThreshold', 200, 'NumOctaves', 6, 'NumScaleLevels', 10);

        [f1, vpts1] = extractFeatures(IrefGray, pts1);
        [f2, vpts2] = extractFeatures(I2gray,    pts2);
        idxPairs    = matchFeatures(f1, f2, 'Unique', true);
        if isempty(idxPairs)
            error('NoMatches');
        end
        matched1 = vpts1(idxPairs(:,1));
        matched2 = vpts2(idxPairs(:,2));

        % Estimate geometric transform
        tforms(k) = estimateGeometricTransform(matched2, matched1, ...
            'similarity', 'MaxDistance',35, ...
            'Confidence',99.9, 'MaxNumTrials',5000);

    catch ME
        % --- Fallback: intensity-based registration ---
        warning('SURF failed for image %d (%s): %s\nFalling back to imregtform.', ...
                k, imageFiles(k).name, ME.message);
        tforms(k) = imregtform(...
            I2gray, IrefGray, ...        % moving, fixed
            'similarity', ...             % settings analogous
            optimizerConfig(), ...
            metricConfig());
    end
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
        Iw = imwarp(Itemp, tforms(k), 'OutputView', outputView);
    else
        Iw = Iorig;
    end
    % Ensure crop rectangle fits within bounds
    [hIw, wIw, ~] = size(Iw);
    xEnd = min(rect(1)+rect(3)-1, wIw);
    yEnd = min(rect(2)+rect(4)-1, hIw);
    cropRect = [rect(1), rect(2), xEnd-rect(1), yEnd-rect(2)];
    Icrop = imcrop(Iw, cropRect);
    if isempty(Icrop)
        warning('The cropped image of "%s" is empty and will not be saved.', imageFiles(k).name);
    else
        imwrite(Icrop, fullfile(croppedDir, imageFiles(k).name));
    end
end

% 10. Comparison and display of matched features (optional)
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
showMatchedFeatures(IrefGray, I2gray, matched1, matched2);
title(sprintf('Residual errors ≤ %d px'));

% 11. Show all common crops individually
cropFiles = dir(fullfile(croppedDir, '*.jpg'));
numCrop   = numel(cropFiles);
for k = 1:numCrop
    I = imread(fullfile(croppedDir, cropFiles(k).name));
    figure('Name', sprintf('Crop: %s', cropFiles(k).name), 'NumberTitle', 'off');
    imshow(I);
    title(cropFiles(k).name, 'Interpreter', 'none');
end

% 12. (Optional) Show all originals individually
for k = 1:numImages
    Iorig = imread(fullfile(folder, imageFiles(k).name));
    figure('Name', sprintf('Original: %s', imageFiles(k).name), 'NumberTitle', 'off');
    imshow(Iorig);
    title(imageFiles(k).name, 'Interpreter', 'none');
end
% 13. SURF-Punkte je Bild einzeln (jeweils Referenz und bewegtes Bild)
for k = 2:numImages
    % Einlesen und Histogramm-Anpassung
    I2RGB     = imread(fullfile(folder, imageFiles(k).name));
    I2matched = histMatchToRef(I2RGB, IrefRGB);
    I2gray    = im2double(rgb2gray(I2matched));
    
    % SURF-Feature-Detektion im ROI
    pts1 = detectSURFFeatures(IrefGray, 'ROI', surfROI, ...
        'MetricThreshold', 200, 'NumOctaves', 6, 'NumScaleLevels', 10);
    pts2 = detectSURFFeatures(I2gray,    'ROI', surfROI, ...
        'MetricThreshold', 200, 'NumOctaves', 6, 'NumScaleLevels', 10);

    % Stärkste 100 Punkte auswählen
    strongest1 = pts1.selectStrongest(100);
    strongest2 = pts2.selectStrongest(100);

    % Figure für Referenzbild
    figure('Name', sprintf('Ref SURF %d', k), 'NumberTitle', 'off');
    imshow(IrefGray);
    hold on;
    plot(strongest1);
    rectangle('Position', surfROI, 'EdgeColor', 'g', 'LineWidth', 1);
    hold off;
    axis off;

    % Figure für aktuelles Bild
    figure('Name', sprintf('Mov SURF %d', k), 'NumberTitle', 'off');
    imshow(I2gray);
    hold on;
    plot(strongest2);
    rectangle('Position', surfROI, 'EdgeColor', 'g', 'LineWidth', 1);
    hold off;
    axis off;
end

%% Helper function: Histogram matching of RGB images
function Iout = histMatchToRef(Iin, Iref)
    Iout = zeros(size(Iin), 'like', Iin);
    for c = 1:size(Iin,3)
        Iout(:,:,c) = imhistmatch(Iin(:,:,c), Iref(:,:,c));
    end
end

%% Helper function: Optimizer configuration for imregtform
function opt = optimizerConfig()
    opt = registration.optimizer.OnePlusOneEvolutionary();
    opt.GrowthFactor      = 1.05;
    opt.InitialRadius     = 0.006;
    opt.Epsilon           = 1.5e-4;
    opt.MaximumIterations = 200;
end

%% Helper function: Metric configuration for imregtform
function m = metricConfig()
    m = registration.metric.MeanSquares();
end
