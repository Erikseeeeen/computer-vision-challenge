%% Main Script: Satellite Image Registration (1400x1000)
clc; clear; close all;

% 1. Directory setup - Ändern Sie den Pfad bei Bedarf
scriptDir = fileparts(mfilename('fullpath'));
folder    = fullfile(scriptDir, 'Datasets', 'Satellite/');

% 2. Gather image files (supporting common formats)
extensions = {'.jpg','.JPG','.tif','.TIF','.png','.PNG'};
imageFiles = [];
for i = 1:length(extensions)
    imageFiles = [imageFiles; dir(fullfile(folder, ['*' extensions{i}]))];
end

if isempty(imageFiles)
    error('Keine Bilder im Ordner "%s" gefunden.', folder);
end
[~, idx] = sort({imageFiles.name});
imageFiles = imageFiles(idx);
numImages = numel(imageFiles);

% 3. Load reference image with size validation
[Iref, map, alpha] = imread(fullfile(folder, imageFiles(1).name));
if ~isempty(map)  % Handle indexed images
    IrefRGB = ind2rgb(Iref, map);
elseif size(Iref, 3) == 1  % Grayscale to RGB
    IrefRGB = cat(3, Iref, Iref, Iref);
else  % Standard RGB
    IrefRGB = Iref;
end
if ~isempty(alpha)  % Apply alpha mask
    IrefRGB = uint8(double(IrefRGB) .* repmat(double(alpha)/255, [1,1,3]));
end

% Validate image size
targetSize = [1000, 1400];  % [height, width]
if any(size(IrefRGB, [1,2]) ~= targetSize)
    warning('Bildgröße ist %dx%d, erwartet 1400x1000. Skalierung wird durchgeführt.',...
        size(IrefRGB,2), size(IrefRGB,1));
    IrefRGB = imresize(IrefRGB, targetSize);
end

% 4. Preprocessing for small images
IrefGray = preprocessImage(IrefRGB, targetSize);
surfROI = [1, 1, targetSize(2), targetSize(1)];  % Entire image

% 5. Initialize transformations
tforms(numImages) = affine2d(eye(3));
tforms(1) = affine2d(eye(3));

% 6. Enhanced registration loop for small images
for k = 2:numImages
    fprintf('Verarbeite %d/%d: %s\n', k, numImages, imageFiles(k).name);
    
    % Load and preprocess
    I2 = imread(fullfile(folder, imageFiles(k).name));
    if any(size(I2, [1,2]) ~= targetSize)
        I2 = imresize(I2, targetSize);
    end
    I2Gray = preprocessImage(I2, targetSize);
    
    % Feature detection optimized for 1400x1000
    pts1 = detectSURFFeatures(IrefGray, 'ROI', surfROI,...
        'MetricThreshold', 300,...  % Lower threshold for more features
        'NumOctaves', 4,...          % Reduced for small images
        'NumScaleLevels', 6);        % Reduced complexity
    
    pts2 = detectSURFFeatures(I2Gray, 'ROI', surfROI,...
        'MetricThreshold', 300,...
        'NumOctaves', 4,...
        'NumScaleLevels', 6);
    
    % Feature extraction
    [f1, vpts1] = extractFeatures(IrefGray, pts1, 'Upright', true);
    [f2, vpts2] = extractFeatures(I2Gray, pts2, 'Upright', true);
    
    % Feature matching with relaxed parameters
    idxPairs = matchFeatures(f1, f2,...
        'Method', 'NearestNeighborRatio',...
        'MatchThreshold', 75,...     % More tolerant (default 100)
        'MaxRatio', 0.7,...           % More matches
        'Unique', true);
    
    if size(idxPairs, 1) < 10
        warning('Nur %d Matches für Bild %d. Verwende verbesserten Fallback.', size(idxPairs,1), k);
        
        % Fallback: Grid-based feature detection
        gridStep = 100;
        [x,y] = meshgrid(100:gridStep:targetSize(2)-100, 100:gridStep:targetSize(1)-100);
        points = cornerPoints([x(:), y(:)]);
        
        pts1 = detectSURFFeatures(IrefGray, 'ROI', surfROI, 'MetricThreshold', 50);
        pts2 = detectSURFFeatures(I2Gray, 'ROI', surfROI, 'MetricThreshold', 50);
        pts1 = pts1.selectStrongest(200);
        pts2 = pts2.selectStrongest(200);
        pts1 = [pts1; points];
        pts2 = [pts2; points];
        
        [f1, vpts1] = extractFeatures(IrefGray, pts1);
        [f2, vpts2] = extractFeatures(I2Gray, pts2);
        idxPairs = matchFeatures(f1, f2, 'MatchThreshold', 40, 'MaxRatio', 0.8);
    end
    
    if size(idxPairs, 1) < 4
        warning('Nur %d Matches - Verwende Identitätstransformation', size(idxPairs,1));
        tforms(k) = affine2d(eye(3));
        continue;
    end
    
    matched1 = vpts1(idxPairs(:,1));
    matched2 = vpts2(idxPairs(:,2));
    
    % RANSAC with pyramid refinement
    for pyramidLevel = 1:2
        scale = 1/(2^(pyramidLevel-1));
        
        if pyramidLevel > 1
            matched1_scaled = matched1.Location * scale;
            matched2_scaled = matched2.Location * scale;
        else
            matched1_scaled = matched1.Location;
            matched2_scaled = matched2.Location;
        end
        
        [tform, inlierIdx] = estimateGeometricTransform(...
            matched2_scaled, matched1_scaled,...
            'similarity',...
            'MaxNumTrials', 3000,...
            'Confidence', 99.5,...
            'MaxDistance', 1.5 + pyramidLevel);
        
        % Refine with inliers only
        if pyramidLevel < 2
            matched1 = matched1(inlierIdx);
            matched2 = matched2(inlierIdx);
        end
    end
    tforms(k) = tform;
end

% 7. Calculate common region using multi-resolution approach
pyramidLevels = 2;
composite = imresize(IrefRGB, 1/(2^(pyramidLevels-1)));
for k = 2:min(3, numImages)  % Use first 3 images
    I2 = imread(fullfile(folder, imageFiles(k).name));
    if any(size(I2, [1,2]) ~= targetSize)
        I2 = imresize(I2, targetSize);
    end
    I2reg = imwarp(I2, tforms(k));
    I2reg_small = imresize(I2reg, 1/(2^(pyramidLevels-1)));
    composite = imfuse(composite, I2reg_small, 'blend', 'Scaling', 'joint');
end

mask = rgb2gray(composite) > 0.05;
mask = imresize(mask, targetSize, 'nearest');
mask = imclose(mask, strel('disk', 15));
stats = regionprops(mask, 'BoundingBox');
rect = stats(1).BoundingBox;

% 8. Save results with metadata
resultsDir = fullfile(folder, 'registered_results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

% Save transformation data
save(fullfile(resultsDir, 'transformations.mat'), 'tforms', 'rect');

% Save cropped images
for k = 1:numImages
    % Load and transform
    I = imread(fullfile(folder, imageFiles(k).name));
    if any(size(I, [1,2]) ~= targetSize)
        I = imresize(I, targetSize);
    end
    
    if k == 1
        Ireg = I;
    else
        Ireg = imwarp(I, tforms(k));
    end
    
    % Crop and save
    Icrop = imcrop(Ireg, rect);
    
    % Preserve filename
    [~,name,ext] = fileparts(imageFiles(k).name);
    imwrite(Icrop, fullfile(resultsDir, [name '_reg' ext]));
    
    % Create preview
    preview = imresize(Icrop, [500, 700]);
    imwrite(preview, fullfile(resultsDir, [name '_preview.jpg']), 'Quality', 85);
end

% Generate report
generateReport(imageFiles, resultsDir, rect);
disp('Verarbeitung erfolgreich abgeschlossen!');

%% Hilfsfunktion: Bildvorverarbeitung für kleine Satellitenbilder
function Iout = preprocessImage(Iin, targetSize)
    % Convert to grayscale
    if size(Iin,3) == 3
        Igray = im2double(rgb2gray(Iin));
    else
        Igray = im2double(Iin);
    end
    
    % Resize if needed
    if nargin > 1 && any(size(Igray) ~= targetSize(1:2))
        Igray = imresize(Igray, targetSize(1:2));
    end
    
    % Optimized enhancement for small images
    Igray = adapthisteq(Igray, 'ClipLimit',0.02, 'Distribution','rayleigh', 'NumTiles',[6 8]);
    Igray = imadjust(Igray, stretchlim(Igray, [0.02 0.98]));
    
    % Gentle sharpening
    Iout = imsharpen(Igray, 'Radius',1, 'Amount',0.8);
end

%% Report generator
function generateReport(imageFiles, resultsDir, rect)
    fig = figure('Visible','off', 'Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    
    % Header
    annotation('textbox',[0.1 0.92 0.8 0.06],...
        'String','Satellitenbild-Registrierungsreport',...
        'FontSize',18, 'FontWeight','bold',...
        'HorizontalAlignment','center', 'LineStyle','none');
    
    % Parameters
    paramStr = sprintf('Bilder: %d\nCrop-Bereich: [x=%d, y=%d, w=%d, h=%d]',...
        numel(imageFiles), round(rect(1)), round(rect(2)), round(rect(3)), round(rect(4)));
    annotation('textbox',[0.1 0.82 0.3 0.1],...
        'String',paramStr,...
        'FontSize',10,...
        'BackgroundColor',[0.95 0.95 0.95]);
    
    % Preview images
    previews = dir(fullfile(resultsDir, '*_preview.jpg'));
    numPreview = min(6, numel(previews));
    ax = gobjects(1, numPreview);
    
    for i = 1:numPreview
        ax(i) = subplot(2,3,i);
        imshow(fullfile(resultsDir, previews(i).name));
        title(previews(i).name, 'Interpreter','none', 'FontSize',8);
    end
    
    % Save report
    reportPath = fullfile(resultsDir, 'registration_report.pdf');
    exportgraphics(fig, reportPath);
    close(fig);
end