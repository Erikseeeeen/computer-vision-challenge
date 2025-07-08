% Skript zur Erkennung linienartiger grauer Strukturen im HSV-Farbraum als Overlay über Originalbilder

% 1. Verzeichnisse festlegen
tScriptDir   = fileparts(mfilename('fullpath'));
inputFolder  = fullfile(tScriptDir, 'Datasets', 'Wiesn');
outputFolder = fullfile(inputFolder, 'Overlays');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% 2. Alle Bilder einlesen
files1     = dir(fullfile(inputFolder, '*.jpg'));
files2     = dir(fullfile(inputFolder, '*.JPG'));
imageFiles = [files1; files2];
if isempty(imageFiles)
    error('Keine Bilder im Ordner "%s" gefunden.', inputFolder);
end
[~, idx]   = sort({imageFiles.name});
imageFiles = imageFiles(idx);
numImages  = numel(imageFiles);

% 3. Parameter (HSV-Schwellen und Morphologie)
satThresh    = 0.2;    % maximale Sättigung für graue Töne
minValThresh = 0.3;    % minimale Helligkeit (Value)
maxValThresh = 0.8;    % maximale Helligkeit (Value)

areaMin      = 1;      % Min. Größe grauer Regionen (Pixel)
opRadius     = 5;      % Opening-Radius (Pixel)
closeRadius  = 1;      % Closing-Radius (Pixel)
distMax      = 200;    % max. Abstand zu farbigen Pixeln (Pixel)

% 4. Hough-Transform-Parameter
theta         = -90:0.1:89.9;  % erweiterte Winkelauflösung
rhoRes        = 1;            % Rho-Auflösung
minLen        = 50;           % minimale Linienlänge (Pixel)
fillGap       = 5;            % Lückenfüllung (Pixel)

% 5. Schleife über alle Bilder
for i = 1:numImages
    % 5.1 Bild einlesen und in HSV konvertieren
    I_rgb = imread(fullfile(inputFolder, imageFiles(i).name));
    if size(I_rgb,3) ~= 3
        I_rgb = cat(3, I_rgb, I_rgb, I_rgb);
    end
    I_hsv = rgb2hsv(I_rgb);
    H = I_hsv(:,:,1);
    S = I_hsv(:,:,2);
    V = I_hsv(:,:,3);

    % 5.2 Maske grauer Bereiche
    BW_gray = (S <= satThresh) & (V >= minValThresh) & (V <= maxValThresh);
    % Farbe ausschließen
    redMask   = (H <= 0.05) | (H >= 0.95);
    greenMask = (H >= 0.25) & (H <= 0.4);
    BW_gray = BW_gray & ~(redMask | greenMask);

    % 5.3 Morphologische Reinigung
    BW_gray = bwareaopen(BW_gray, areaMin);
    BW_gray = imopen(BW_gray, strel('disk', opRadius));
    BW_gray = imclose(BW_gray, strel('disk', closeRadius));
    BW_gray = bwmorph(BW_gray, 'bridge');
    BW_gray = BW_gray & (bwdist(redMask|greenMask) <= distMax);

    % 5.4 Hough-Transformation
    [Hacc, T, R] = hough(BW_gray, 'RhoResolution', rhoRes, 'Theta', theta);
    peaks        = houghpeaks(Hacc, 150, 'Threshold', ceil(0.3 * max(Hacc(:))));
    lines        = houghlines(BW_gray, T, R, peaks, 'MinLength', minLen, 'FillGap', fillGap);

    % 5.5 Kürzeste 30% Linien verwerfen
    if numel(lines) > 1
        lengths = arrayfun(@(ln) norm(ln.point2 - ln.point1), lines);
        thr     = prctile(lengths, 30);
        lines   = lines(lengths > thr);
    end

    % 5.6 Overlay generieren und speichern
    % Zeichne Linien unsichtbar und exportiere direkt in ein Bild
    hFig = figure('Visible', 'off');
    imshow(I_rgb); hold on;
    for k = 1:numel(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'r-', 'LineWidth', 2);
    end
    hold off;
    % Bild aus Figur extrahieren
    frame       = getframe(gca);
    overlayImg  = frame2im(frame);
    close(hFig);

    % Filename mit Suffix
    [~, name, ext] = fileparts(imageFiles(i).name);
    outName        = sprintf('%s_overlay%s', name, ext);
    imwrite(overlayImg, fullfile(outputFolder, outName));
end

% Ende des Skripts
