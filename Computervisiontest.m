% -------------------------------------------------------------------------
% Vollständiges Skript: Veränderungen zwischen jeweils aufeinanderfolgenden Bildern
% inklusive Histogram Matching zur Belichtungskompensation
% -------------------------------------------------------------------------

% 1. Ordnerpfad anpassen
ordner = 'C:\Users\jhall\OneDrive\Desktop\Computervision\Datasets\Brazilian Rainforest';

% 2. Alle JPG-Dateien im Ordner einlesen und nach Name sortieren
files = dir(fullfile(ordner, '*.jpg'));
namen = {files.name};
[~, sortIdx] = sort(namen);
files = files(sortIdx);
numImages = numel(files);

% 3. Referenzbild laden (RGB)
IrefRGB = imread(fullfile(ordner, files(1).name));

% 4. Schleife über alle Bildpaare k → k+1
for k = 1:numImages-1
    % --- 4.1 Bilder laden ---
    I1RGB = imread(fullfile(ordner, files(k).name));
    I2RGB = imread(fullfile(ordner, files(k+1).name));
    
    % --- 4.2 Histogram Matching: Belichtung von I2 an I1 anpassen ---
    I2matched = zeros(size(I2RGB), 'like', I2RGB);
    for c = 1:3
        I2matched(:,:,c) = imhistmatch(I2RGB(:,:,c), I1RGB(:,:,c));
    end
    
    % --- 4.3 In Graustufen und double konvertieren ---
    I1 = im2double(rgb2gray(I1RGB));
    I2 = im2double(rgb2gray(I2matched));
    
    % --- 4.4 Feature-Matching für Registrierung (SURF) ---
    pts1 = detectSURFFeatures(I1);
    pts2 = detectSURFFeatures(I2);
    [f1, vpts1] = extractFeatures(I1, pts1);
    [f2, vpts2] = extractFeatures(I2, pts2);
    idxPairs = matchFeatures(f1, f2, 'Unique', true);
    matched1 = vpts1(idxPairs(:,1));
    matched2 = vpts2(idxPairs(:,2));
    
    % --- 4.5 Similarity-Transformation schätzen (Rotation & Skalierung) ---
    tic
    tform = estimateGeometricTransform(matched2, matched1, ...
        'similarity', 'MaxDistance', 4, 'Confidence', 99.9, 'MaxNumTrials', 2000);
    toc
    % --- 4.6 Warp des angeglichenen Bildes auf das Koordinatensystem des ersten ---
    outputView = imref2d(size(I1));
    I2regGray = imwarp(I2,   tform, 'OutputView', outputView);
    I2regRGB  = imwarp(I2matched, tform, 'OutputView', outputView);
    
    % --- 4.7 Gemeinsamen Ausschnitt bestimmen ---
    mask = I2regGray > 0;
    stats = regionprops(mask, 'Area', 'BoundingBox');
    [~, idxMax] = max([stats.Area]);
    bb = stats(idxMax).BoundingBox;  % [x, y, Breite, Höhe]
    x1 = floor(bb(1)) + 1;
    y1 = floor(bb(2)) + 1;
    w  = floor(bb(3));
    h  = floor(bb(4));
    
    % --- 4.8 Zuschneiden beider Bilder ---
    I1c = I1RGB(y1:y1+h-1,   x1:x1+w-1, :);
    I2c = I2regRGB(y1:y1+h-1, x1:x1+w-1, :);
    
    % --- 4.9 Veränderungsmaske und Overlay erstellen ---
    G1 = im2double(rgb2gray(I1c));
    G2 = im2double(rgb2gray(I2c));
    diff      = abs(G2 - G1);
    schwelle  = 0.2;
    changeMask = diff > schwelle;
    redMask   = uint8(cat(3, changeMask, zeros(size(changeMask)), zeros(size(changeMask)))) * 255;
    overlay   = imadd(I2c, redMask);
    
    % --- 4.10 Ergebnisse anzeigen ---
    figure('Name', sprintf('Vergleich %d → %d', k, k+1), 'NumberTitle', 'off');
    subplot(1,3,1), imshow(I1c), title(['Ref: ' files(k).name], 'Interpreter','none');
    subplot(1,3,2), imshow(I2c), title(['Mov (matched): ' files(k+1).name], 'Interpreter','none');
    subplot(1,3,3), imshow(overlay), title('Veränderungen (rot)', 'Interpreter','none');
    
    % --- Optional: Overlay exportieren ---
    % outName = fullfile(ordner, sprintf('change_%02d_%02d.png', k, k+1));
    % imwrite(overlay, outName);
end
