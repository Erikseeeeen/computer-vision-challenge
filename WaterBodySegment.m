function waterSegmentationResults = WaterBodySegment(imagePath)
% segmentWater - Water segmentation on an image.

    % Input Validation and Image Loading
    if ~exist(imagePath, 'file')
        error('segmentWater:FileNotFound', 'Image file not found at path: %s', imagePath);
    end
    img = imread(imagePath);
    img_double = im2double(img);
    [rows, cols, ~] = size(img_double);
    total_pixels = rows * cols;

    R = img_double(:,:,1);
    G = img_double(:,:,2);
    B = img_double(:,:,3);

    % METHOD 1: Modified Normalized Difference Water Index (MNDWI)
    [water_mndwi, mndwi_index] = calculateMNDWI(G, R);

    % METHOD 2: Blue-Green Water Index
    [water_bg, blue_green_index] = calculateBlueGreenIndex(B, G, R);

    % METHOD 3: Automated Water Extraction Index (AWEI)
    [water_awei, awei_index] = calculateAWEI(G, R);

    % METHOD 4: Otsu's Automatic Thresholding on Water-Sensitive Channels
    water_otsu_blue = applyOtsuThreshold(B);
    mndwi_norm = (mndwi_index + 1) / 2;
    water_otsu_mndwi = applyOtsuThreshold(mndwi_norm);

    % METHOD 5: HSV-based Water Detection
    water_hsv = detectWaterHSV(img_double);

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

    water_ensemble = ensemble_score > 0.5;

    % POST-PROCESSING: Morphological Operations
    final_water_mask = postProcessWaterMask(water_ensemble);

    pixel_area = sum(final_water_mask(:));
    water_percentage = (pixel_area / total_pixels) * 100;

    % Spatial calibratio
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

    waterSegmentationResults = struct();
    waterSegmentationResults.originalImage = img;
    waterSegmentationResults.finalWaterMask = final_water_mask;

    % overlay = img;
    % blue_channel_overlay = overlay(:,:,3);
    % blue_channel_overlay(final_water_mask) = 255;
    % overlay(:,:,3) = blue_channel_overlay;

    blueChannel  = uint8(final_water_mask) * 255;
    waterSegmentationResults.overlayAlpha = blueChannel;
    
    overlay = zeros(size(img), 'uint8');
    overlay(:,:,3) = blueChannel; % Blue channel
    waterSegmentationResults.overlay = overlay;

    waterSegmentationResults.overlayImage = overlay;

    waterSegmentationResults.confidenceMap = ensemble_score;

    % Store individual method masks and index maps
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

    % Store statistics
    stats = struct();
    stats.pixelArea = pixel_area;
    stats.totalPixels = total_pixels;
    stats.waterPercentage = water_percentage;
    stats.realWorldArea = real_world_area;
    stats.avgBoundaryStrength = avg_boundary_strength;
    stats.numWaterRegions = num_water_regions;
    stats.waterRegionProps = water_stats_props;
    stats.mndwiWaterPercentage = (sum(water_mndwi(:))/total_pixels)*100;
    stats.aweiWaterPercentage = (sum(water_awei(:))/total_pixels)*100;
    stats.blueGreenWaterPercentage = (sum(water_bg(:))/total_pixels)*100;
    stats.otsuBlueWaterPercentage = (sum(water_otsu_blue(:))/total_pixels)*100;
    stats.hsvWaterPercentage = (sum(water_hsv(:))/total_pixels)*100;

    waterSegmentationResults.stats = stats;
end


function [water_mask, mndwi] = calculateMNDWI(green_channel, red_channel)
% calculateMNDWI - This module calculates the Modified Normalized Difference Water Index (MNDWI)
%   using the green and red channels of an image.
    denominator = green_channel + red_channel;
    valid_mask = denominator > 0.01;
    mndwi = zeros(size(green_channel), 'double');
    mndwi(valid_mask) = (green_channel(valid_mask) - red_channel(valid_mask)) ./ denominator(valid_mask);
    water_mask = mndwi > 0;
end

function [water_mask, bg_index] = calculateBlueGreenIndex(blue_channel, green_channel, red_channel)
% calculateBlueGreenIndex - This module computes a custom Blue-Green Water Index,
%   effective for detecting water based on the relative intensities of blue, green, and red.
    bg_index = (blue_channel + green_channel) - red_channel;
    water_mask = bg_index > 0.1;
end

function [water_mask, awei_index] = calculateAWEI(green_channel, red_channel)
% calculateAWEI - This module approximates the Automated Water Extraction Index (AWEI),
%   useful for robust water detection using the green and red channels.
    awei_index = 4 * (green_channel - red_channel) - 0.25 * red_channel - 2.75 * red_channel;
    water_mask = awei_index > 0;
end

function water_mask = applyOtsuThreshold(channel_data)
% applyOtsuThreshold - This module applies Otsu's method to automatically determine
%   and apply a threshold to a given image channel to generate a binary mask.
    if max(channel_data(:)) > 1 || min(channel_data(:)) < 0
        warning('applyOtsuThreshold:Normalization', 'Input data for Otsu not normalized to [0,1]. Normalizing now.');
        channel_data = (channel_data - min(channel_data(:))) / (max(channel_data(:)) - min(channel_data(:)));
    end
    level = graythresh(channel_data);
    water_mask = channel_data > level;
end

function water_mask = detectWaterHSV(img_double)
% detectWaterHSV - This module identifies water regions based on specific
%   hue, saturation, and value (HSV) ranges in the image.
    hsv = rgb2hsv(img_double);
    H = hsv(:,:,1);
    S = hsv(:,:,2);
    V = hsv(:,:,3);

    water_mask = ((H >= 0.4 & H <= 0.8) | (H >= 0.0 & H <= 0.1)) & ...
                 (S >= 0.02) & ...
                 (V >= 0.1 & V <= 0.9);
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