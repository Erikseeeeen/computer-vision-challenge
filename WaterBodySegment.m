function waterSegmentationResults = WaterBodySegment(imagePath, varargin)
% WaterBodySegment - Water segmentation on an image with configurable method and sensitivity.
%
% Syntax: 
%   results = WaterBodySegment(imagePath)
%   results = WaterBodySegment(imagePath, 'Method', method, 'Sensitivity', sensitivity)
%
% Inputs:
%   imagePath - Path to the input image
%   Method - (optional) String specifying segmentation method:
%            'MNDWI', 'Blue-Green', 'AWEI', 'OTSU', 'HSV', or 'Ensemble' (default)
%   Sensitivity - (optional) Numeric value controlling detection sensitivity
%                Default values vary by method

    % Parse input arguments
    p = inputParser;
    addRequired(p, 'imagePath', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Method', 'Ensemble', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Sensitivity', [], @(x) isnumeric(x) && isscalar(x));
    
    parse(p, imagePath, varargin{:});
    
    method = upper(string(p.Results.Method));
    sensitivity = p.Results.Sensitivity;
    
    % Validate method
    validMethods = ["MNDWI", "BLUE-GREEN", "AWEI", "OTSU", "HSV", "ENSEMBLE"];
    if ~ismember(method, validMethods)
        error('WaterBodySegment:InvalidMethod', ...
              'Method must be one of: %s', strjoin(validMethods, ', '));
    end
    
    % Set default sensitivity values for each method
    defaultSensitivity = containers.Map();
    defaultSensitivity('MNDWI') = 0.0;
    defaultSensitivity('BLUE-GREEN') = 0.1;
    defaultSensitivity('AWEI') = 0.0;
    defaultSensitivity('OTSU') = 0.0;  % Offset from Otsu threshold
    defaultSensitivity('HSV') = 0.0;   % Range expansion factor
    defaultSensitivity('ENSEMBLE') = 0.5;  % Ensemble threshold
    
    if isempty(sensitivity)
        sensitivity = defaultSensitivity(char(method));
    end

    % Input Validation and Image Loading
    if ~exist(imagePath, 'file')
        error('WaterBodySegment:FileNotFound', 'Image file not found at path: %s', imagePath);
    end
    img = imread(imagePath);
    img_double = im2double(img);
    [rows, cols, ~] = size(img_double);
    total_pixels = rows * cols;

    R = img_double(:,:,1);
    G = img_double(:,:,2);
    B = img_double(:,:,3);

    % Execute selected method or ensemble
    switch method
        case 'MNDWI'
            [final_water_mask, method_index] = calculateMNDWI(G, R, sensitivity);
            confidence_map = double(final_water_mask);
            
        case 'BLUE-GREEN'
            [final_water_mask, method_index] = calculateBlueGreenIndex(B, G, R, sensitivity);
            confidence_map = double(final_water_mask);
            
        case 'AWEI'
            [final_water_mask, method_index] = calculateAWEI(G, R, sensitivity);
            confidence_map = double(final_water_mask);
            
        case 'OTSU'
            final_water_mask = applyOtsuThreshold(B, sensitivity);
            method_index = B;
            confidence_map = double(final_water_mask);
            
        case 'HSV'
            final_water_mask = detectWaterHSV(img_double, sensitivity);
            method_index = rgb2hsv(img_double);
            confidence_map = double(final_water_mask);
            
        case 'ENSEMBLE'
            % Calculate all methods with their default sensitivities for ensemble
            [water_mndwi, mndwi_index] = calculateMNDWI(G, R, defaultSensitivity('MNDWI'));
            [water_bg, blue_green_index] = calculateBlueGreenIndex(B, G, R, defaultSensitivity('BLUE-GREEN'));
            [water_awei, awei_index] = calculateAWEI(G, R, defaultSensitivity('AWEI'));
            water_otsu_blue = applyOtsuThreshold(B, defaultSensitivity('OTSU'));
            mndwi_norm = (mndwi_index + 1) / 2;
            water_otsu_mndwi = applyOtsuThreshold(mndwi_norm, defaultSensitivity('OTSU'));
            water_hsv = detectWaterHSV(img_double, defaultSensitivity('HSV'));

            % ENSEMBLE METHOD: Combine Multiple Approaches
            weights.mndwi = 0.3;
            weights.awei = 0.25;
            weights.bg = 0.2;
            weights.otsu_blue = 0.15;
            weights.hsv = 0.1;

            ensemble_score = weights.mndwi * double(water_mndwi) + ...
                             weights.awei * double(water_awei) + ...
                             weights.bg * double(water_bg) + ...
                             weights.otsu_blue * double(water_otsu_blue) + ...
                             weights.hsv * double(water_hsv);

            final_water_mask = ensemble_score > sensitivity;
            confidence_map = ensemble_score;
            method_index = ensemble_score;
            
        otherwise
            error('WaterBodySegment:UnexpectedMethod', 'Unexpected method: %s', method);
    end

    % POST-PROCESSING: Morphological Operations (only for single methods, not ensemble)
    if ~strcmp(method, 'ENSEMBLE')
        final_water_mask = postProcessWaterMask(final_water_mask);
    end

    pixel_area = sum(final_water_mask(:));
    water_percentage = (pixel_area / total_pixels) * 100;

    % Spatial calibration
    scale_bar_pixels = 50;
    real_world_length = 10;
    conversion_factor = (real_world_length / scale_bar_pixels)^2;
    real_world_area = pixel_area * conversion_factor;

    gray_img = rgb2gray(img_double);
    [Gx, Gy] = gradient(gray_img);
    gradient_mag = sqrt(Gx.^2 + Gy.^2);
    boundary_pixels = bwperim(final_water_mask);
    avg_boundary_strength = mean(gradient_mag(boundary_pixels));
    if isempty(boundary_pixels) || isnan(avg_boundary_strength)
        avg_boundary_strength = 0;
    end

    % Water body statistics
    water_stats_props = regionprops(final_water_mask, 'Area', 'Perimeter', 'Solidity', 'BoundingBox');
    num_water_regions = length(water_stats_props);

    % Build results structure
    waterSegmentationResults = struct();
    waterSegmentationResults.originalImage = img;
    waterSegmentationResults.finalWaterMask = final_water_mask;

    blueChannel = uint8(final_water_mask) * 255;
    waterSegmentationResults.overlayAlpha = blueChannel;
    
    overlay = zeros(size(img), 'uint8');
    overlay(:,:,3) = blueChannel; % Blue channel
    waterSegmentationResults.overlay = overlay;
    waterSegmentationResults.overlayImage = overlay;
    waterSegmentationResults.confidenceMap = confidence_map;

    % Store method-specific results
    waterSegmentationResults.selectedMethod = char(method);
    waterSegmentationResults.sensitivity = sensitivity;
    waterSegmentationResults.methodIndexMap = method_index;

    % Store statistics
    stats = struct();
    stats.pixelArea = pixel_area;
    stats.totalPixels = total_pixels;
    stats.waterPercentage = water_percentage;
    stats.realWorldArea = real_world_area;
    stats.avgBoundaryStrength = avg_boundary_strength;
    stats.numWaterRegions = num_water_regions;
    stats.waterRegionProps = water_stats_props;
    stats.selectedMethod = char(method);
    stats.sensitivity = sensitivity;

    waterSegmentationResults.stats = stats;
    
    % If ensemble was used, also store individual method results
    if strcmp(method, 'ENSEMBLE')
        waterSegmentationResults.mndwiMask = water_mndwi;
        waterSegmentationResults.mndwiIndexMap = mndwi_index;
        waterSegmentationResults.aweiMask = water_awei;
        waterSegmentationResults.aweiIndexMap = awei_index;
        waterSegmentationResults.blueGreenMask = water_bg;
        waterSegmentationResults.blueGreenIndexMap = blue_green_index;
        waterSegmentationResults.otsuBlueMask = water_otsu_blue;
        waterSegmentationResults.otsuMNDWIMask = water_otsu_mndwi;
        waterSegmentationResults.hsvMask = water_hsv;
        waterSegmentationResults.ensembleScoreMap = ensemble_score;
        
        % Add individual method statistics
        stats.mndwiWaterPercentage = (sum(water_mndwi(:))/total_pixels)*100;
        stats.aweiWaterPercentage = (sum(water_awei(:))/total_pixels)*100;
        stats.blueGreenWaterPercentage = (sum(water_bg(:))/total_pixels)*100;
        stats.otsuBlueWaterPercentage = (sum(water_otsu_blue(:))/total_pixels)*100;
        stats.hsvWaterPercentage = (sum(water_hsv(:))/total_pixels)*100;
        waterSegmentationResults.stats = stats;
    end
end


function [water_mask, mndwi] = calculateMNDWI(green_channel, red_channel, sensitivity)
% calculateMNDWI - This module calculates the Modified Normalized Difference Water Index (MNDWI)
%   using the green and red channels of an image with configurable sensitivity.
    if nargin < 3
        sensitivity = 0.0;  % Default threshold
    end
    
    denominator = green_channel + red_channel;
    valid_mask = denominator > 0.01;
    mndwi = zeros(size(green_channel), 'double');
    mndwi(valid_mask) = (green_channel(valid_mask) - red_channel(valid_mask)) ./ denominator(valid_mask);
    water_mask = mndwi > sensitivity;
end

function [water_mask, bg_index] = calculateBlueGreenIndex(blue_channel, green_channel, red_channel, sensitivity)
% calculateBlueGreenIndex - This module computes a custom Blue-Green Water Index,
%   effective for detecting water based on the relative intensities of blue, green, and red.
    if nargin < 4
        sensitivity = 0.1;  % Default threshold
    end
    
    bg_index = (blue_channel + green_channel) - red_channel;
    water_mask = bg_index > sensitivity;
end

function [water_mask, awei_index] = calculateAWEI(green_channel, red_channel, sensitivity)
% calculateAWEI - This module approximates the Automated Water Extraction Index (AWEI),
%   useful for robust water detection using the green and red channels.
    if nargin < 3
        sensitivity = 0.0;  % Default threshold
    end
    
    awei_index = 4 * (green_channel - red_channel) - 0.25 * red_channel - 2.75 * red_channel;
    water_mask = awei_index > sensitivity;
end

function water_mask = applyOtsuThreshold(channel_data, sensitivity)
% applyOtsuThreshold - This module applies Otsu's method to automatically determine
%   and apply a threshold to a given image channel to generate a binary mask.
%   Sensitivity parameter adjusts the threshold relative to Otsu's automatic value.
    if nargin < 2
        sensitivity = 0.0;  % Default: no adjustment to Otsu threshold
    end
    
    if max(channel_data(:)) > 1 || min(channel_data(:)) < 0
        warning('applyOtsuThreshold:Normalization', 'Input data for Otsu not normalized to [0,1]. Normalizing now.');
        channel_data = (channel_data - min(channel_data(:))) / (max(channel_data(:)) - min(channel_data(:)));
    end
    level = graythresh(channel_data);
    adjusted_level = level + sensitivity;  % Adjust threshold by sensitivity
    adjusted_level = max(0, min(1, adjusted_level));  % Clamp to [0,1]
    water_mask = channel_data > adjusted_level;
end

function water_mask = detectWaterHSV(img_double, sensitivity)
% detectWaterHSV - This module identifies water regions based on specific
%   hue, saturation, and value (HSV) ranges in the image.
%   Sensitivity parameter expands or contracts the detection ranges.
    if nargin < 2
        sensitivity = 0.0;  % Default: no range adjustment
    end
    
    hsv = rgb2hsv(img_double);
    H = hsv(:,:,1);
    S = hsv(:,:,2);
    V = hsv(:,:,3);

    % Base ranges
    hue_range1_low = 0.4;
    hue_range1_high = 0.8;
    hue_range2_low = 0.0;
    hue_range2_high = 0.1;
    sat_low = 0.02;
    val_low = 0.1;
    val_high = 0.9;
    
    % Adjust ranges based on sensitivity
    % Positive sensitivity expands ranges, negative contracts them
    hue_expansion = sensitivity * 0.1;  % Scale factor for hue range expansion
    sat_adjustment = sensitivity * 0.01; % Scale factor for saturation adjustment
    val_adjustment = sensitivity * 0.05; % Scale factor for value adjustment
    
    % Apply adjustments (with bounds checking)
    hue_range1_low = max(0, hue_range1_low - hue_expansion);
    hue_range1_high = min(1, hue_range1_high + hue_expansion);
    hue_range2_high = min(1, hue_range2_high + hue_expansion);
    sat_low = max(0, sat_low - sat_adjustment);
    val_low = max(0, val_low - val_adjustment);
    val_high = min(1, val_high + val_adjustment);

    water_mask = ((H >= hue_range1_low & H <= hue_range1_high) | (H >= hue_range2_low & H <= hue_range2_high)) & ...
                 (S >= sat_low) & ...
                 (V >= val_low & V <= val_high);
end

function cleaned_mask = postProcessWaterMask(input_mask)
% postProcessWaterMask - This module refines a binary water mask using
%   morphological operations to remove noise, fill holes, and smooth boundaries.
    cleaned_mask = bwareaopen(input_mask, 200);
    cleaned_mask = imfill(cleaned_mask, 'holes');
    se_smooth = strel('disk', 2);
    cleaned_mask = imopen(cleaned_mask, se_smooth);
    cleaned_mask = imclose(cleaned_mask, se_smooth);
    cleaned_mask = bwareaopen(cleaned_mask, 500);
end