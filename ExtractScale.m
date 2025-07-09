function m_per_pixel = ExtractScale(currentLocation)
    % Extracts the scale from an image located in a specified folder.
    % Reads an image from the specified subfolder, processes it to identify the scale bar,
    % and calculates the scale in meters per pixel.
    %
    % Inputs:
    %   currentLocation - The subfolder name where the image is stored.


    % Load the image from the specified path
    image = fullfile(pwd, 'Datasets', currentLocation);
    I = imread(image);

    % Get the dimensions of the image
    [h, w, ~] = size(I);

    % Crop the bottom 10% in height and right 20% in width
    roi = I(round(0.90*h):h, round(0.80*w):w, :);

    % Convert the cropped region to grayscale
    G = rgb2gray(roi);

    % Perform OCR to extract text from the image
    results = ocr(roi);
    scaleText = strtrim(results.Text);  % e.g. scaleText = "200 m"

    % Detect edges in the grayscale image
    edges = edge(G, 'Canny');

    % Perform Hough Transform to detect lines
    [H, theta, rho] = hough(edges);
    peaks = houghpeaks(H, 5, 'Threshold', ceil(0.3 * max(H(:))));
    lines = houghlines(edges, theta, rho, peaks, 'FillGap', 5, 'MinLength', 20);

    [hR, wR] = size(edges);  
    candidates = [];  % Indices of good lines

    % Identify near-horizontal lines in the lower-right region
    for k = 1:length(lines)
        L = lines(k);
        if abs(L.theta) < 5  % Check if the line is near horizontal
            mid = (L.point1 + L.point2) / 2;  % Compute midpoint
            if mid(1) > 0.5 * wR && mid(2) > 0.5 * hR  % Restrict to lower-right half
                candidates(end + 1) = k;  %#ok<AGROW>
            end
        end
    end

    % Determine pixel length of the scale bar
    if isempty(candidates)
        warning('No bottom-right horizontal lines found; falling back to longest overall.');
        pixelLength = max(arrayfun(@(L) norm(L.point1 - L.point2), lines));
    else
        % Compute lengths of candidate lines
        lens = arrayfun(@(idx) norm(lines(idx).point1 - lines(idx).point2, candidates);
        [pixelLength, mi] = max(lens);
        bestLine = lines(candidates(mi));
        px1 = bestLine.point1;
        px2 = bestLine.point2;
    end

    % Display the scale bar pixel length
    %fprintf('Scale-bar pixel length: %.1f px\n', pixelLength);

    % Extract numerical value and unit from scale text
    tokens = regexp(scaleText, '([\d\.]+)\s*(m|km)', 'tokens');
    val = str2double(tokens{1}{1});
    unit = tokens{1}{2};

    % Convert to meters
    meters = val * (strcmp(unit, 'km') * 1000 + strcmp(unit, 'm') * 1);
    m_per_pixel = meters / pixelLength;  % Calculate scale in m/pixel

    % Display the calculated scale
    %Sfprintf('Scale: %.3f m/pixel\n', m_per_pixel);
end